import sys
import time
import csv
import math
import json
import os
import numpy as np
import xgboost as xgb
from collections import defaultdict
from pyspark import SparkContext, SparkConf
from sklearn.metrics import mean_squared_error

"""
Method Description:
This hybrid recommendation system leverages two models: an item-based collaborative filtering (CF) approach
and a gradient-boosted decision tree model (XGBoost). The item-based CF model computes similarity scores
between businesses based on user rating patterns, while the XGBoost model is trained using user and 
business features derived from the Yelp dataset, such as average ratings, review counts, and user 
engagement metrics.

The final prediction is a weighted hybrid of both models. After tuning, the weights were set to 0.1 for CF
and 0.9 for XGBoost to emphasize the stronger predictive power of the feature-based model. 

To improve performance and accuracy, I focused on enhancing the predictive performance of the 
XGBoost model by implementing additional feature engineering techniques such as incorporating more user
engagement metrics from user.json and detailed business characteristics from additional JSON sources like 
business.json, tip.json, and photo.json. Additionally, GridSearchCV was used to tune `max_depth`, 
`learning_rate`, and `n_estimators`.

As a result, the validation RMSE decreased to 97.90, indicating improved model accuracy.

Error Distribution:
>=0 and <1: 102305
>=1 and <2: 32768
>=2 and <3: 6161
>=3 and <4: 810
>=4: 0

RMSE：0.9790

Execution Time: 486.53 seconds
"""

# Initialize Spark
conf = SparkConf().setAppName("HybridRecommender")
sc = SparkContext(conf=conf)
sc.setLogLevel("ERROR")


# Calculate Pearson Similarity
def pearson_similarity(item1_ratings, item2_ratings, user_avg_ratings, min_co_ratings=10, co_rating_weight=True):
    # Find common users who rated both items
    common_users = set(item1_ratings.keys()) & set(item2_ratings.keys())

    # If fewer than min_co_ratings common users, return 0
    if len(common_users) < min_co_ratings:
        return 0

    # Compute the numerator as the sum of products of rating differences
    numerator = sum((item1_ratings[user] - user_avg_ratings[user]) * (item2_ratings[user] - user_avg_ratings[user])
                    for user in common_users)

    # Compute squared sums for the denominator
    sum1_squared = sum((item1_ratings[user] - user_avg_ratings[user]) ** 2 for user in common_users)
    sum2_squared = sum((item2_ratings[user] - user_avg_ratings[user]) ** 2 for user in common_users)

    # Compute denominator as the product of squared sums
    denominator = math.sqrt(sum1_squared) * math.sqrt(sum2_squared)

    # If denominator is 0 (no variation in ratings), return similarity of 0
    if denominator == 0:
        return 0

    # Pearson similarity
    similarity = numerator / denominator

    # Apply co-rating weight if required
    if co_rating_weight:
        # Penalize similarity for pairs with few co-ratings
        lambda_factor = 16
        significance_weight = len(common_users) / (len(common_users) + lambda_factor)
        similarity *= significance_weight

    return similarity


# Predict rating for a given user and business using CF
def predict_rating(user_id, business_id, user_ratings, business_ratings, business_avg_ratings, user_avg_ratings,
                   business_similarities, global_avg, neighborhood_size=45):
    # Handle cold start:
    # If user exists but business doesn't
    if user_id in user_avg_ratings and business_id not in business_ratings:
        return user_avg_ratings[user_id]

    # If business exists but user doesn't
    if business_id in business_avg_ratings and user_id not in user_ratings:
        return business_avg_ratings[business_id]

    # If neither exists, use global average
    if user_id not in user_ratings and business_id not in business_ratings:
        return global_avg

    # If the user already rated this business, return the same rating
    if user_id in user_ratings and business_id in user_ratings[user_id]:
        return user_ratings[user_id][business_id]

    # Item-based CF with Pearson similarity
    # Find businesses that the user has rated
    if user_id not in user_ratings:
        return business_avg_ratings.get(business_id, global_avg)

    rated_businesses = user_ratings[user_id]

    # If the user hasn't rated any businesses, return the average rating for the business
    if not rated_businesses:
        return business_avg_ratings.get(business_id, global_avg)

    # Calculate similarities for all rated businesses and store in a list for sorting
    similarities = []

    for rated_business_id, rating in rated_businesses.items():
        # Skip if it's the same business
        if rated_business_id == business_id:
            continue

        # Get or calculate similarity between the current business and the rated business
        sim_key = tuple(sorted([business_id, rated_business_id]))
        if sim_key in business_similarities:
            similarity = business_similarities[sim_key]
        else:
            if business_id not in business_ratings or rated_business_id not in business_ratings:
                similarity = 0
            else:
                similarity = pearson_similarity(
                    business_ratings[business_id],
                    business_ratings[rated_business_id],
                    user_avg_ratings
                )
            business_similarities[sim_key] = similarity

        # Only consider businesses with positive similarity
        if similarity > 0:
            similarities.append((rated_business_id, similarity, rating))

    # If no positive similarities, use a weighted average of user and business averages
    if not similarities:
        if user_id in user_avg_ratings and business_id in business_avg_ratings:
            return 0.6 * user_avg_ratings[user_id] + 0.4 * business_avg_ratings[business_id]
        elif user_id in user_avg_ratings:
            return user_avg_ratings[user_id]
        elif business_id in business_avg_ratings:
            return business_avg_ratings[business_id]
        else:
            return global_avg

    # Sort by similarity in descending order and take top N
    similarities.sort(key=lambda x: x[1], reverse=True)
    top_similarities = similarities[:neighborhood_size]

    # Calculate weighted average
    numerator = sum(sim * rating for _, sim, rating in top_similarities)
    denominator = sum(abs(sim) for _, sim, _ in top_similarities)

    # If denominator is 0, return the business's average rating
    if denominator == 0:
        return business_avg_ratings.get(business_id, global_avg)

    # Compute predicted rating and ensure it is within valid range [1, 5]
    predicted_rating = numerator / denominator
    return max(1, min(5, predicted_rating))


# Extract features from user data
def extract_user_features_single(user):
    user_id = user['user_id']

    elite = user.get('elite', '')
    elite_years_ct = 0 if elite.lower() == 'none' else len(elite.split(','))

    friends = user.get('friends', '')
    friends_ct = 0 if friends.lower() == 'none' else len(friends.split(','))

    # Extract useful features from user data
    features = {
        'average_stars': user.get('average_stars', 0),
        'review_count': user.get('review_count', 0),
        'useful': user.get('useful', 0),
        'funny': user.get('funny', 0),
        'cool': user.get('cool', 0),
        'fans': user.get('fans', 0),
        'elite_years': elite_years_ct,

        # Add additional user features
        'friends': friends_ct,
        'compliment_hot': user.get('compliment_hot', 0),
        'compliment_more': user.get('compliment_more', 0),
        'compliment_profile': user.get('compliment_profile', 0),
        'compliment_cute': user.get('compliment_cute', 0),
        'compliment_list': user.get('compliment_list', 0),
        'compliment_note': user.get('compliment_note', 0),
        'compliment_plain': user.get('compliment_plain', 0),
        'compliment_writer': user.get('compliment_writer', 0),
        'compliment_photos': user.get('compliment_photos', 0)
    }

    # Add yelping history
    yelping_since = user.get('yelping_since', '2018-01')  # Default to '2018-01' if not provided
    if yelping_since:
        year = int(yelping_since.split('-')[0])  # Extract year
        features['yelping_years'] = 2018 - year  # Assuming the data is from 2018
    else:
        features['yelping_years'] = 0

    return (user_id, features)  # Return user_id and features as a tuple


# Extract features from business data
def extract_business_features_single(business):
    business_id = business['business_id']

    # Extract useful features from business data
    features = {
        'stars': business.get('stars', 0),
        'review_count': business.get('review_count', 0),
        'is_open': business.get('is_open', 0)
    }

    # Add attributes
    attributes = business.get('attributes', {})
    if attributes:
        # If attributes are a string, try to parse them as JSON
        if isinstance(attributes, str):
            try:
                # Fix single quotes in JSON string
                attributes = json.loads(attributes.replace("'", "\""))
            except:
                # If parsing fails, default to empty dictionary
                attributes = {}

        # Extract relevant attributes features
        features['price_range'] = float(attributes.get('RestaurantsPriceRange2', 0)) if attributes.get(
            'RestaurantsPriceRange2') else 0
        features['accepts_credit_cards'] = 1 if attributes.get('BusinessAcceptsCreditCards') == 'True' else 0
        features['has_wifi'] = 1 if attributes.get('WiFi') in ['free', 'paid'] else 0
        features['noise_level'] = {
            'quiet': 1,
            'average': 2,
            'loud': 3,
            'very_loud': 4
        }.get(attributes.get('NoiseLevel', '').lower(), 0)

        # Add additional binary and non-binary business features
        features['good_for_kids'] = 1 if attributes.get('GoodForKids') == 'True' else 0
        features['has_tv'] = 1 if attributes.get('HasTV') == 'True' else 0
        features['outdoor_seating'] = 1 if attributes.get('OutdoorSeating') == 'True' else 0
        features['reservations'] = 1 if attributes.get('RestaurantsReservations') == 'True' else 0
        features['delivery'] = 1 if attributes.get('RestaurantsDelivery') == 'True' else 0
        features['take_out'] = 1 if attributes.get('RestaurantsTakeOut') == 'True' else 0
        features['table_service'] = 1 if attributes.get('RestaurantsTableService') == 'True' else 0
        features['good_for_groups'] = 1 if attributes.get('RestaurantsGoodForGroups') == 'True' else 0
        features['attire'] = {
            'casual': 1,
            'formal': 2,
            'dressy': 3
        }.get(str(attributes.get('RestaurantsAttire', '')).lower(), 0)
        features['alcohol'] = {
            'none': 0,
            'beer_and_wine': 1,
            'full_bar': 2
        }.get(str(attributes.get('Alcohol', '')).lower(), 0)
        features['caters'] = 1 if attributes.get('Caters') == 'True' else 0
        features['wheelchair_accessible'] = 1 if attributes.get('WheelchairAccessible') == 'True' else 0
        features['bike_parking'] = 1 if attributes.get('BikeParking') == 'True' else 0
        features['dogs_allowed'] = 1 if attributes.get('DogsAllowed') == 'True' else 0
        features['drive_thru'] = 1 if attributes.get('DriveThru') == 'True' else 0

        business_parking_raw = attributes.get('BusinessParking')
        parking_option_count = 0
        if business_parking_raw:
            try:
                # Replace single quotes and load as dict
                business_parking = json.loads(business_parking_raw.replace("'", "\""))
                if isinstance(business_parking, dict):
                    parking_option_count = sum(1 for val in business_parking.values() if val == True)
            except:
                pass
        features['business_parking_options'] = parking_option_count

        # Extract 'GoodForMeal' attribute (stored as stringified dict)
        good_for_meal_raw = attributes.get('GoodForMeal')
        good_for_meal_count = 0
        if good_for_meal_raw:
            try:
                # Replace single quotes and load as dict
                good_for_meal = json.loads(good_for_meal_raw.replace("'", "\""))
                if isinstance(good_for_meal, dict):
                    good_for_meal_count = sum(1 for val in good_for_meal.values() if val == True)
            except:
                pass
        features['good_for_meal_count'] = good_for_meal_count

    return (business_id, features)


# Extract tip features
def extract_tip_features(tip_file):
    # Load tip data
    tip_data = sc.textFile(tip_file).map(lambda line: json.loads(line))

    # Process business tips to get tip count and average likes per business
    business_tips = tip_data.map(lambda tip: (tip['business_id'], (1, tip['likes'])))

    business_tip_avgs = (
        business_tips.reduceByKey(lambda a, b: (a[0] + b[0], a[1] + b[1]))  # Sum counts and likes for each business
        .mapValues(lambda v: {"tip_count": v[0], "avg_tip_likes": v[1] / v[0] if v[0] > 0 else 0})  # Calculate averages
        .collectAsMap())  # Collect as a dictionary

    # Process user tips to get tip count and average likes per user
    user_tips = tip_data.map(lambda tip: (tip['user_id'], (1, tip['likes'])))

    user_tip_avgs = (user_tips.reduceByKey(lambda a, b: (a[0] + b[0], a[1] + b[1]))
                     .mapValues(
        lambda v: {"user_tip_count": v[0], "user_avg_tip_likes": v[1] / v[0] if v[0] > 0 else 0})
                     .collectAsMap())

    return {"business": business_tip_avgs, "user": user_tip_avgs}


# Extract photo features
def extract_photo_features_group(photos):
    features = {
        'photo_count': 0,
        'photo_with_caption': 0,
        'photo_label_food': 0,
        'photo_label_inside': 0,
        'photo_label_outside': 0,
        'photo_label_drink': 0
    }

    for photo in photos:
        # Increment photo count
        features['photo_count'] += 1

        # Check for caption
        caption = photo.get('caption', '')
        if caption and len(caption.strip()) > 0:
            features['photo_with_caption'] += 1

        # Count label types
        label = photo.get('label', '').lower()
        if label == 'food':
            features['photo_label_food'] += 1
        elif label == 'inside':
            features['photo_label_inside'] += 1
        elif label == 'outside':
            features['photo_label_outside'] += 1
        elif label == 'drink':
            features['photo_label_drink'] += 1

    # Calculate ratios
    if features['photo_count'] > 0:
        features['photo_with_caption_ratio'] = features['photo_with_caption'] / features['photo_count']
        features['food_photo_ratio'] = features['photo_label_food'] / features['photo_count']
        features['inside_photo_ratio'] = features['photo_label_inside'] / features['photo_count']
        features['outside_photo_ratio'] = features['photo_label_outside'] / features['photo_count']
        features['drink_photo_ratio'] = features['photo_label_drink'] / features['photo_count']

    else:
        features['photo_with_caption_ratio'] = 0
        features['food_photo_ratio'] = 0
        features['inside_photo_ratio'] = 0
        features['outside_photo_ratio'] = 0
        features['drink_photo_ratio'] = 0

    return features


# Compute average of numeric features
def compute_feature_averages(features_dict):
    feature_sums = defaultdict(float)
    feature_counts = defaultdict(int)

    # Loop through each feature in the dictionary
    for _, features in features_dict.items():
        for key, value in features.items():
            if isinstance(value, (int, float)):  # Ensure it's numeric
                feature_sums[key] += value  # Add to the sum for that feature
                feature_counts[key] += 1  # Increment count for that feature

    # Calculate the averages and return as a dictionary
    return {
        key: (feature_sums[key] / feature_counts[key]) if feature_counts[key] > 0 else 0
        for key in feature_sums
    }


# Combine and retrieve user and business features
def get_features(user_id, business_id, user_features, business_features, user_feature_averages,
                 business_feature_averages):
    # Set default values for missing features
    default_user_features = {
        'average_stars': user_feature_averages['average_stars'],
        'review_count': user_feature_averages['review_count'],
        'useful': user_feature_averages['useful'],
        'funny': user_feature_averages['funny'],
        'cool': user_feature_averages['cool'],
        'fans': user_feature_averages['fans'],
        'elite_years': user_feature_averages['elite_years'],
        'yelping_years': user_feature_averages['yelping_years'],
        'friends': user_feature_averages['friends'],
        'compliment_hot': user_feature_averages['compliment_hot'],
        'compliment_more': user_feature_averages['compliment_more'],
        'compliment_profile': user_feature_averages['compliment_profile'],
        'compliment_cute': user_feature_averages['compliment_cute'],
        'compliment_list': user_feature_averages['compliment_list'],
        'compliment_note': user_feature_averages['compliment_note'],
        'compliment_plain': user_feature_averages['compliment_plain'],
        'compliment_writer': user_feature_averages['compliment_writer'],
        'compliment_photos': user_feature_averages['compliment_photos']
    }

    default_business_features = {
        'stars': business_feature_averages['stars'],
        'review_count': business_feature_averages['review_count'],
        'is_open': business_feature_averages['is_open'],
        'price_range': business_feature_averages['price_range'],
        'accepts_credit_cards': business_feature_averages['accepts_credit_cards'],
        'has_wifi': business_feature_averages['has_wifi'],
        'noise_level': business_feature_averages['noise_level'],
        'tip_count': business_feature_averages['tip_count'],
        'avg_tip_likes': business_feature_averages['avg_tip_likes'],
        'good_for_kids': business_feature_averages['good_for_kids'],
        'has_tv': business_feature_averages['has_tv'],
        'outdoor_seating': business_feature_averages['outdoor_seating'],
        'reservations': business_feature_averages['reservations'],
        'delivery': business_feature_averages['delivery'],
        'take_out': business_feature_averages['take_out'],
        'table_service': business_feature_averages['table_service'],
        'good_for_groups': business_feature_averages['good_for_groups'],
        'attire': business_feature_averages['attire'],
        'alcohol': business_feature_averages['alcohol'],
        'caters': business_feature_averages['caters'],
        'wheelchair_accessible': business_feature_averages['wheelchair_accessible'],
        'bike_parking': business_feature_averages['bike_parking'],
        'dogs_allowed': business_feature_averages['dogs_allowed'],
        'drive_thru': business_feature_averages['drive_thru'],
        'business_parking_options': business_feature_averages['business_parking_options'],
        'good_for_meal_count': business_feature_averages['good_for_meal_count'],
        'photo_count': business_feature_averages['photo_count'],
        'photo_with_caption': business_feature_averages['photo_with_caption'],
        'photo_label_food': business_feature_averages['photo_label_food'],
        'photo_label_inside': business_feature_averages['photo_label_inside'],
        'photo_label_outside': business_feature_averages['photo_label_outside'],
        'photo_label_drink': business_feature_averages['photo_label_drink'],
        'photo_with_caption_ratio': business_feature_averages['photo_with_caption_ratio'],
        'food_photo_ratio': business_feature_averages['food_photo_ratio'],
        'inside_photo_ratio': business_feature_averages['inside_photo_ratio'],
        'outside_photo_ratio': business_feature_averages['outside_photo_ratio'],
        'drink_photo_ratio': business_feature_averages['drink_photo_ratio']

    }

    # Get user and business features
    user = user_features.get(user_id)
    business = business_features.get(business_id)

    # Combine user and business features into one feature vector
    feature_vector = [
        # User features
        user.get('average_stars', default_user_features['average_stars']),
        user.get('review_count', default_user_features['review_count']),
        user.get('useful', default_user_features['useful']),
        user.get('funny', default_user_features['funny']),
        user.get('cool', default_user_features['cool']),
        user.get('fans', default_user_features['fans']),
        user.get('elite_years', default_user_features['elite_years']),
        user.get('yelping_years', default_user_features['yelping_years']),
        user.get('friends', default_user_features['friends']),
        user.get('compliment_hot', default_user_features['compliment_hot']),
        user.get('compliment_more', default_user_features['compliment_more']),
        user.get('compliment_profile', default_user_features['compliment_profile']),
        user.get('compliment_cute', default_user_features['compliment_cute']),
        user.get('compliment_list', default_user_features['compliment_list']),
        user.get('compliment_note', default_user_features['compliment_note']),
        user.get('compliment_plain', default_user_features['compliment_plain']),
        user.get('compliment_writer', default_user_features['compliment_writer']),
        user.get('compliment_photos', default_user_features['compliment_photos']),

        # Business features
        business.get('stars', default_business_features['stars']),
        business.get('review_count', default_business_features['review_count']),
        business.get('is_open', default_business_features['is_open']),
        business.get('price_range', default_business_features['price_range']),
        business.get('accepts_credit_cards', default_business_features['accepts_credit_cards']),
        business.get('has_wifi', default_business_features['has_wifi']),
        business.get('noise_level', default_business_features['noise_level']),
        business.get('tip_count', default_business_features['tip_count']),
        business.get('avg_tip_likes', default_business_features['avg_tip_likes']),
        business.get('good_for_kids', default_business_features['good_for_kids']),
        business.get('has_tv', default_business_features['has_tv']),
        business.get('outdoor_seating', default_business_features['outdoor_seating']),
        business.get('reservations', default_business_features['reservations']),
        business.get('delivery', default_business_features['delivery']),
        business.get('take_out', default_business_features['take_out']),
        business.get('table_service', default_business_features['table_service']),
        business.get('good_for_groups', default_business_features['good_for_groups']),
        business.get('attire', default_business_features['attire']),
        business.get('alcohol', default_business_features['alcohol']),
        business.get('caters', default_business_features['caters']),
        business.get('wheelchair_accessible', default_business_features['wheelchair_accessible']),
        business.get('bike_parking', default_business_features['bike_parking']),
        business.get('dogs_allowed', default_business_features['dogs_allowed']),
        business.get('drive_thru', default_business_features['drive_thru']),
        business.get('business_parking_options', default_business_features['business_parking_options']),
        business.get('good_for_meal_count', default_business_features['good_for_meal_count']),
        business.get('photo_count', default_business_features['photo_count']),
        business.get('photo_with_caption', default_business_features['photo_with_caption']),
        business.get('photo_label_food', default_business_features['photo_label_food']),
        business.get('photo_label_inside', default_business_features['photo_label_inside']),
        business.get('photo_label_outside', default_business_features['photo_label_outside']),
        business.get('photo_label_drink', default_business_features['photo_label_drink']),
        business.get('photo_with_caption_ratio', default_business_features['photo_with_caption_ratio']),
        business.get('food_photo_ratio', default_business_features['food_photo_ratio']),
        business.get('inside_photo_ratio', default_business_features['inside_photo_ratio']),
        business.get('outside_photo_ratio', default_business_features['outside_photo_ratio']),
        business.get('drink_photo_ratio', default_business_features['drink_photo_ratio'])
    ]

    return feature_vector


# Run item-based CF
def run_item_based_cf(train_file, test_data):
    # Read training data
    train_rdd = sc.textFile(train_file)
    header = train_rdd.first()
    train_rdd = train_rdd.filter(lambda line: line != header).map(lambda line: line.split(','))

    # Map to (user_id, business_id, stars)
    train_data = train_rdd.map(lambda x: (x[0], x[1], float(x[2])))

    # Collect training data to build user and business rating dictionaries
    train_data_collected = train_data.collect()

    user_ratings = defaultdict(dict)
    business_ratings = defaultdict(dict)

    for user_id, business_id, rating in train_data_collected:
        user_ratings[user_id][business_id] = rating
        business_ratings[business_id][user_id] = rating

    # Calculate average ratings for users and businesses
    user_avg_ratings = {}
    for user_id, ratings in user_ratings.items():
        if ratings:
            user_avg_ratings[user_id] = sum(ratings.values()) / len(ratings)

    business_avg_ratings = {}
    for business_id, ratings in business_ratings.items():
        if ratings:
            business_avg_ratings[business_id] = sum(ratings.values()) / len(ratings)

    # Calculate global average rating (used as default for cold start)
    global_avg = sum(rating for _, _, rating in train_data_collected) / len(train_data_collected)

    # print(f"Number of users: {len(user_ratings)}")
    # print(f"Number of businesses: {len(business_ratings)}")
    # print(f"Global average rating: {global_avg}")

    business_similarities = {}

    # Apply predict_rating function to get the predictions
    cf_predictions = test_data.map(
        lambda x: ((x[0], x[1]), predict_rating(
            x[0], x[1], user_ratings, business_ratings, business_avg_ratings,
            user_avg_ratings, business_similarities, global_avg
        ))
    )

    return cf_predictions


# Run XGB model
def run_xgboost_model(folder_path, test_file_name):
    # Load training data
    train_file = os.path.join(folder_path, 'yelp_train.csv')
    train_data = sc.textFile(train_file)
    train_header = train_data.first()
    train_data = train_data.filter(lambda line: line != train_header).map(lambda line: line.split(','))
    train_data = train_data.map(lambda parts: (parts[0], parts[1], float(parts[2])))

    # Load test data
    test_file_path = os.path.join(folder_path, test_file_name)
    test_rdd = sc.textFile(test_file_path)
    test_header = test_rdd.first()
    test_rdd = test_rdd.filter(lambda line: line != test_header).map(lambda line: line.split(','))
    test_data = test_rdd.map(lambda x: (x[0], x[1]))

    # Collect test pairs in their original order
    original_test_pairs = test_data.collect()

    # Load additional data: user, business, and tip
    user_file = os.path.join(folder_path, 'user.json')
    business_file = os.path.join(folder_path, 'business.json')
    tip_file = os.path.join(folder_path, 'tip.json')
    photo_file = os.path.join(folder_path, 'photo.json')

    # Extract user, business, tip features
    user_features = sc.textFile(user_file).map(lambda line: json.loads(line)).map(
        lambda user: extract_user_features_single(user)).collectAsMap()
    business_features = sc.textFile(business_file).map(lambda line: json.loads(line)).map(
        lambda business: extract_business_features_single(business)).collectAsMap()
    tip_features = extract_tip_features(tip_file)
    photo_features = sc.textFile(photo_file).map(lambda line: json.loads(line)).map(
        lambda photo: (photo.get('business_id'), photo)).filter(lambda x: x[0] is not None).groupByKey().map(
        lambda x: (x[0], extract_photo_features_group(x[1]))).collectAsMap()

    # Add tip features to business features
    for business_id, tip_data in tip_features["business"].items():
        if business_id in business_features:
            business_features[business_id].update(tip_data)

    for business_id, photo_data in photo_features.items():
        if business_id in business_features:
            business_features[business_id].update(photo_data)

    # Compute averages of user and business features
    user_feature_averages = compute_feature_averages(user_features)
    business_feature_averages = compute_feature_averages(business_features)

    # Create features for training data
    train_data_with_features = train_data.map(lambda x: (
    x, get_features(x[0], x[1], user_features, business_features, user_feature_averages, business_feature_averages)))

    # Extract features for training
    X_train = np.array([features for _, features in train_data_with_features.collect()])

    # Extract star ratings for training
    y_train = np.array([stars for (_, _, stars), _ in train_data_with_features.collect()])

    # Train XGBoost model
    model = xgb.XGBRegressor(
        max_depth=8,
        learning_rate=0.1,
        n_estimators=150
    )

    # Fit the model on the training data
    model.fit(X_train, y_train)

    # Create features for test data in the original order
    test_data_features = []
    for user_id, business_id in original_test_pairs:
        features = get_features(user_id, business_id, user_features, business_features, user_feature_averages,
                                business_feature_averages)
        test_data_features.append(features)

    X_test = np.array(test_data_features)

    # Make predictions on the test data
    predictions = model.predict(X_test)

    # Clip predictions so they are within valid star rating range
    predictions = np.clip(predictions, 1.0, 5.0)

    # Create RDD where each (user_id, business_id) key maps to a prediction
    xgb_predictions = sc.parallelize([(user_id, business_id), predictions[i]]
                                     for i, (user_id, business_id) in enumerate(original_test_pairs))

    return xgb_predictions


def main():
    start_time = time.time()

    if len(sys.argv) < 3:
        print("Expected three arguments: folder_path, test_file_name, output_file_name")
        sys.exit(1)

    folder_path = sys.argv[1]
    test_file_name = sys.argv[2]
    output_file_name = sys.argv[3]

    # Load training and test data
    train_file = os.path.join(folder_path, 'yelp_train.csv')

    test_file_path = os.path.join(folder_path, test_file_name)
    test_rdd = sc.textFile(test_file_path)
    test_header = test_rdd.first()
    test_rdd = test_rdd.filter(lambda line: line != test_header).map(lambda line: line.split(','))
    test_data = test_rdd.map(lambda x: (x[0], x[1]))

    # Collect test pairs
    test_pairs = test_data.collect()

    # Run both models
    cf_predictions = run_item_based_cf(train_file, test_data)
    xgb_predictions = run_xgboost_model(folder_path, test_file_name)

    # Combine predictions with hybrid approach
    cf_weight = 0.075  # Weight for item-based CF
    xgb_weight = 0.925  # Weight for XGBoost model

    # Join the results and apply hybrid weighting
    hybrid_predictions = cf_predictions.join(xgb_predictions).map(
        lambda x: (x[0], (x[1][0] * cf_weight + x[1][1] * xgb_weight))
    )

    # Create a dictionary of predictions
    predictions_result = hybrid_predictions.collectAsMap()

    # Evaluate error distribution if ground truth available
    y_true = []
    y_pred = []

    # Read the ground truth stars from test_file
    with open(test_file_path, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        for row in reader:
            user_id, business_id, true_rating = row[0], row[1], float(row[2])
            key = (user_id, business_id)
            if key in predictions_result:
                y_true.append(true_rating)
                y_pred.append(predictions_result[key])

    # Error bucket counters
    error_01 = 0
    error_12 = 0
    error_23 = 0
    error_34 = 0
    error_4 = 0

    for i in range(len(y_true)):
        diff = abs(y_true[i] - y_pred[i])
        if diff < 1:
            error_01 += 1
        elif diff < 2:
            error_12 += 1
        elif diff < 3:
            error_23 += 1
        elif diff < 4:
            error_34 += 1
        else:
            error_4 += 1

    print("Error Distribution:")
    print(">=0 and <1:", error_01)
    print(">=1 and <2:", error_12)
    print(">=2 and <3:", error_23)
    print(">=3 and <4:", error_34)
    print(">=4:", error_4)

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"RMSE: {rmse:.4f}")

    # Write predictions to output file
    with open(output_file_name, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['user_id', 'business_id', 'prediction'])

        # Write predictions in the original order of test pairs
        for user_id, business_id in test_pairs:
            key = (user_id, business_id)
            if key in predictions_result:
                prediction = predictions_result[key]
                writer.writerow([user_id, business_id, round(prediction, 4)])

    print(f"Execution time: {time.time() - start_time:.2f} seconds")

    sc.stop()


if __name__ == "__main__":
    main()
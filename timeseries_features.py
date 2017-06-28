import numpy as np
import pandas as pd



import gc


print('loading prior')
priors = pd.read_csv('./data/order_products__prior.csv')

train = pd.read_csv('./data/train_new.csv')
train_eval = pd.read_csv('./data/train_eval.csv')

print('loading orders')
orders = pd.read_csv('./data/orders.csv')


###
# some memory measures for kaggle kernel
print('optimize memory')
orders.order_dow = orders.order_dow.astype(np.int8)
orders.order_hour_of_day = orders.order_hour_of_day.astype(np.int8)
orders.order_number = orders.order_number.astype(np.int16)
orders.order_id = orders.order_id.astype(np.int32)
orders.user_id = orders.user_id.astype(np.int32)
orders.days_since_prior_order = orders.days_since_prior_order.astype(np.float32)


train.reordered = train.reordered.astype(np.int8)
train.add_to_cart_order = train.add_to_cart_order.astype(np.int16)

train_eval.reordered = train.reordered.astype(np.int8)
train_eval.add_to_cart_order = train.add_to_cart_order.astype(np.int16)

priors.order_id = priors.order_id.astype(np.int32)
priors.add_to_cart_order = priors.add_to_cart_order.astype(np.int16)
priors.reordered = priors.reordered.astype(np.int8)
priors.product_id = priors.product_id.astype(np.int32)

gc.collect()


print('Join orders with prior')
orders.set_index('order_id', inplace = True, drop = False)
priors.set_index('order_id', inplace = True, drop = False)

priors = priors.join(orders, on = 'order_id', rsuffix = '_')
priors.drop('order_id_', inplace = True, axis = 1)

priors.reset_index(inplace=True, drop=True)
orders.reset_index(inplace=True, drop=True)


print('Prepare prior for featrues')

prior_subset = pd.DataFrame()

prior_subset['p_count']    = priors.groupby(['user_id','order_id','order_number'], group_keys = False)['product_id'].agg('count')
prior_subset['rorder_sum'] = priors.groupby(['user_id','order_id','order_number'], group_keys = False)['reordered'].agg('sum')
prior_subset['rorder_rate'] = prior_subset['rorder_sum'] / prior_subset['p_count']




prior_subset.reset_index(drop = False, inplace=True)
prior_subset.sort_values(by = ['user_id','order_number'], inplace = True, ascending = [True, True], axis = 0)
prior_subset.drop('order_id', inplace = True, axis = 1)


prior_subset.reset_index(inplace = True, drop = True)

print('generate features')

from tsfresh import extract_features, extract_relevant_features,select_features
from tsfresh.feature_extraction.settings import MinimalFCParameters
#settings = MinimalFCParameters()
#extracted_features = extract_features(prior_subset, column_id="user_id", column_sort = "order_number" ,default_fc_parameters=settings)

extracted_features = extract_features(prior_subset, column_id="user_id", column_sort = "order_number" )

extracted_features.to_csv('./features/ts_features_all.csv', index = False)













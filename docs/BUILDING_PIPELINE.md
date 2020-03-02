# Building Custom Pipelines
Before diving into this, you should check out the [query-level and market-level scorer sections](MULBERRY.md#computing-the-fitness-of-a-child-)
## Table of Contents
1. [Generating the data](#generating-the-data)
2. [Creating a scoring config](#creating-a-scoring-config)
3. [Training a model](#training-a-model)
4. [Evaluating the model offline](#evaluating-the-model-offline)

## Generating the data
#### Filter down to relevant requests
If you plan to just optimize for something like purchase-NDCG, you should filter your requests down to just ones containing purchases first.
### Splitting your data
It's best to provide separate train and validation files. We recommend splitting your training data into train and validation, then testing on the next day of data. 
### Formatting your data
Before you can use this framework, you need to convert your data into libSVM format.

The expected format of one record is (space separated):
```
relevance_score qid:id metadata1:val1 metadata2:val2 metadata3:val3 feature1:f1 feature2:f2 feature3:f3
```

Where metadata values can either be String or Float values.

_Note: Currently we do not support boolean metadata. If you provide this, it will automatically be treated as a string and may result in some unexpected behavior._

## Creating a scoring config
You will need to create config that specifies the query-level metrics to compute and how to aggregate them at the market level. This config will also specify what policy to use (e.g. pointwise, beam).

You will control the hyperparameter weights for the different optimization goals through this config as well. For example, if you want to optimize for avg(purchase-NDCG@10) and median(avg-price@10) across purchase requests, you would use the following query level config:
```json
[
    {
        "name": "avg-price-10",
        "group": "avg-price-10",
        "weight": 1.0,
        "scorer": {
            "Mean": {
                "k": 10,
                "field_name": "price"
            }
        }
    },
    {
        "name": "purchase-ndcg",
        "group": "purchase-ndcg",
        "weight": 1.0,
        "scorer": {
            "NDCG": {
                "k": 10
            }
        }
    }
]
```
and the following indicator level config:
```json
[
    {
        "name": "avg-price-q1",
        "weight": 0.2,
        "indicator": {
            "Histogram": ["avg-price-10", [25]]
        },
        "filter_config": {
            "MetricGreaterThan": ["is_purchase_request", 0.0]
        }
    },
    {
        "name": "relevance",
        "weight": 0.8,
        "indicator": {
            "Histogram": ["purchase-ndcg", [50]]
        },
        "filter_config": {
            "MetricGreaterThan": ["is_purchase_request", 0.0]
        }
    }
]
```

This sets up two query level scorers to compute NDCG and avg(price@10). Using those scorers, we compute the 25th quantile of avg-price-10 across all requests where is_purchase_request is greater than 0. This also computes the 50th quantile of purchase-NDCG. These are combined to generate the fitness value.

### Purpose of the query-level weight and group.
Before we combine a metric across requests, you have the option to combine different metrics at the query level. For example, this could be useful if you have is_purchase_request in one metadata field and is_mobile in another, but want to filter down to just the purchase requests on mobile when computing the market place metric. If you don't plan to combine metrics within a query first, you should provide the same name and group along with `weight = 1.0`.

## Training a model
_Note: Since libSVM is 1-indexed but Mulberry is 0-indexed, you'll actually need to provide features + 1 when specifying the number of features_

Once you've generated some data in the appropriate format and have made your scoring config, you can now train a model!
You'll need to [install Evokit on your local machine](docs/DEVELOPMENT.md#installing-locally) first.
```bash
mulberry data/train \
-v data/valid \
--features 2 \
--scoring-config configs/example.json \
--save-model /tmp/model/linear/example \
simple
```
The above command trains a linear model with 1 feature using the simple optimizer. This command would output logs such as:

```
Dims: 2
Loading dataset at data/train
Query sets: 80
Examples: 3523
Reading validation dataset data/valid...
Query sets: 80
Examples: 3523
Loading scoring config: configs/example.json
Num scorers added: 5
Num indicators added: 3
Num scorers added: 5
Num indicators added: 3
Num scorers added: 29
Num indicators added: 187
Time: 0.000,	Iteration: 0,	Fitness: 0.2380592,	Valid: 0.2380592,	Stats: Some([("opt-buyer-promise", 0.8668418), ("opt-price-purchases", 1.0), ("opt-relevance-purchases", 0.2380592)])
Time: 0.012,	Iteration: 1,	Fitness: 0.25502113,	Valid: 0.25502113,	Stats: Some([("opt-buyer-promise", 0.9046504), ("opt-price-purchases", 1.0), ("opt-relevance-purchases", 0.25502113)])
...
Time: 0.038,	Iteration: 100,	Fitness: 0.38283855,	Valid: 0.38283855,	Stats: Some([("opt-buyer-promise", 0.90079236), ("opt-price-purchases", 1.0), ("opt-relevance-purchases", 0.38283855)])
Valid fitness: 0.38283855
Valid Logger: Some([("opt-buyer-promise", 0.90079236), ("opt-price-purchases", 1.0), ("opt-relevance-purchases", 0.38283855)])
Writing model to /tmp/model/linear/example
```

## Evaluating the model offline
Based on the metrics and indicators defined in the scoring config, you can evaluate test data using Mulberry as well. You will need to provide a path to the saved model file and the scoring config to evaluate the model.
```bash
mulberry-test \
--features 2 \
--model-path model_path \
--scoring-config scoring_config \
--test test_data
```

This computes all the indicators defined in the scoring config. A sample output can be seen below:

```
Dims: 2
Loading scoring config: configs/example.json
Num scorers added: 5
Num indicators added: 3
Num scorers added: 5
Num indicators added: 3
Num scorers added: 29
Num indicators added: 187
Testing against data/test
Test Fitness: 0.38283855
Test Logger: Some([("avg-is-bp-24-mean", 0.90079236), ("avg-is-bp-24-q1", 0.90909094), ("avg-is-bp-24-q2", 1.0), ("avg-is-bp-24-q3", 1.0), ("avg-price-1-mean", 36.799995), ("avg-price-1-purchases-mean", 36.799995), ("avg-price-1-purchases-q1", 7.62), ("avg-price-1-purchases-q2", 14.969999), ("avg-price-1-purchases-q3", 33.0), ..., ("opt-relevance-purchases", 0.38283855)])
```

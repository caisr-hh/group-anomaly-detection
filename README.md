# COSMO
Self-Monitoring with a Group-based Anomaly Detection Approach.

This is an implementation a group-based anomaly detection method. It allows to autonomously monitor a system (unit) that generates data over time, by comparing it against a group of other similar systems (units). It detects anomalies/deviations in a streaming fashion while accounting for concept drift which is due to external factors that can affect the data.

First, you need to import `SelfMonitoring` and create an instance:
```python
from selfmonitoring import SelfMonitoring

# Create an instance of SelfMonitoring
sm = SelfMonitoring(w_ref_group="7days",        # Time window for the reference group
                    w_martingale=15,            # Window size for computing the deviation level
                    non_conformity="median",    # Non-conformity (strangeness) measure: "median" or "knn"
                    k=50,                       # Used if non_conformity is "knn"
                    dev_threshold=.6)           # Threshold on the deviation level
```

An example data is provided in this repository under the folder ./data. It contains data from 19 units (vehicles). Each csv file corresponds to the data from one unit. To use the example data provided in this repository, you can import `Data` and load it as folows:
```python
from data import Data

# Simulates data generation from several units at each day ("1d")
dataset = Data("./data/").load()
```

A streaming setting is considered where data (indicated as `x_units` in the exemple below) is received from the units at each time step `dt`. The method `SelfMonitoring.diagnoise(uid, dt, x_units)` is then called each time to diagnoise the test unit indicated by the index `uid` (i.e. the data-point received from this unit at time `dt` is `x_units[uid]`). The `diagnoise` method returns:
1. a *strangeness* score : the non-conformity of the test unit to the other units).
2. a *p-value* (in [0, 1]) : the proportion of data from other units which are stranger than the test unit's data.
3. an updated *devaliation* level (in [0, 1]) for the test unit.
4. a boolean *is_dev* indicating if the test unit is significantly deviating from the group.
```python
'''At each time dt, x_units contains data from all units.
Each data-point x_units[i] comes from the i'th unit.'''

for dt, x_units in dataset.getNext():
    # diagnoise the selected test unit (at index 0)
    strangeness, pvalue, deviation, is_dev = sm.diagnoise(0, dt, x_units)
    
    print("Time: {} ==> strangeness: {}, p-value: {}, deviation: {} ({})"
        .format(dt, strangeness, pvalue, deviation, "high" if is_dev else "low"))
```

The deviation level for the test unit can be plotted as folows
```python
# Plot p-values and deviation level over time
sm.plot_deviation()
```

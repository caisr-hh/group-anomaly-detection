from cosmo.datasets import load_vehicles
from cosmo.single_anomaly import InductiveAnomaly

# Get data from one unit (vehicle)
dataset = load_vehicles()
unit0 = [ x for dt, x in dataset.stream_unit(0) ]
unit0_train = unit0[len(unit0)//2:]
unit0_test = unit0[:len(unit0)//2]

# Create an instance of InductiveAnomaly
ina = InductiveAnomaly( w_martingale=15,            # Window size for computing the deviation level
                        non_conformity="median",    # Non-conformity (strangeness) measure: "median" or "knn"
                        k=50,                       # Used if non_conformity is "knn"
                        dev_threshold=.6)           # Threshold on the deviation level

# Fit the InductiveAnomaly detector to unit0_train
ina.fit(unit0_train)

# At each time step, a new test data-point x comes from unit0_test
for i, x in enumerate(unit0_test):
    # diagnoise unit0
    strangeness, pvalue, deviation, is_dev = ina.predict(x)
    
    print("Stream: {} ==> strangeness: {}, p-value: {}, deviation: {} ({})"
        .format(i, strangeness, pvalue, deviation, "high" if is_dev else "low"))

# Plot p-values and deviation level over time
ina.plot_deviation()

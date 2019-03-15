from cosmo.datasets import load_vehicles
from cosmo.group_anomaly import GroupAnomaly

# Streams data from several units (vehicles) over time
dataset = load_vehicles()

# Create an instance of GroupAnomaly
ga = GroupAnomaly(  w_ref_group="7days",        # Time window for the reference group
                    w_martingale=15,            # Window size for computing the deviation level
                    non_conformity="median",    # Non-conformity (strangeness) measure: "median" or "knn"
                    k=50,                       # Used if non_conformity is "knn"
                    dev_threshold=.6)           # Threshold on the deviation level

# At each time dt, a data-point x_units[i] comes from the i'th unit
for dt, x_units in dataset.stream():
    # diagnoise the selected test unit (at index 0)
    strangeness, pvalue, deviation, is_dev = ga.predict(0, dt, x_units)
    
    print("Time: {} ==> strangeness: {}, p-value: {}, deviation: {} ({})"
        .format(dt, strangeness, pvalue, deviation, "high" if is_dev else "low"))

# Plot p-values and deviation level over time
ga.plot_deviation()

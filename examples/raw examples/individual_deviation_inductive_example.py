from grand.datasets import load_vehicles, load_artificial
from grand import IndividualAnomalyInductive

if __name__ == '__main__':
    
    # Get data from one unit (vehicle)
    dataset = load_artificial(0) #load_vehicles()
    unit1_train = [ x for dt, x in dataset.stream_unit(1) ] # we use unit number 1 for training

    # Create an instance of IndividualAnomalyInductive
    indev = IndividualAnomalyInductive( w_martingale=15,            # Window size for computing the deviation level
                                        non_conformity="median",    # Strangeness measure: "median" or "knn" or "lof"
                                        k=50,                       # Used if non_conformity is "knn"
                                        dev_threshold=.6)           # Threshold on the deviation level

    # Fit the IndividualAnomalyInductive detector to unit1_train
    indev.fit(unit1_train)

    # At each time step dt, a data-point x comes from the stream of unit number 0
    for dt, x in dataset.stream_unit(0):
        devContext = indev.predict(dt, x)
        
        st, pv, dev, isdev = devContext.strangeness, devContext.pvalue, devContext.deviation, devContext.is_deviating
        print("Time: {} ==> strangeness: {}, p-value: {}, deviation: {} ({})".format(dt, st, pv, dev, "high" if isdev else "low"))

    # Plot p-values and deviation level over time
    indev.plot_deviations(figsize=(10,5))
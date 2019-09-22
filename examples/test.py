from grand import IndividualAnomalyTransductive, datasets

dataset = datasets.load_artificial(0)
data0 = list(dataset.stream_unit(0))
data1 = list(dataset.stream_unit(1))

model = IndividualAnomalyTransductive()

for t, x in data1:
    info = model.predict(t, x)
    print(info)

model.plot_deviations()

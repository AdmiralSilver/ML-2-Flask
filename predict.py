import joblib
import numpy as np

model = joblib.load('ML-2/models/nocompressmodel.joblib')


def get_form_data(data):
    values = {
        'budget': 0,
        'popularity': 0,
        'runtime': 0,
        'cast_amount': 0,
        'crew_amount': 0,
    }
    for key in [k for k in data.keys() if k in values.keys()]:
        values[key] = data[key]
    return values


def predict(data, debug=False):
    values = get_form_data(data)

    if debug:
        print(f'values: {values}\n')

    column_order = ['budget', 'popularity', 'runtime', 'cast_amount', 'crew_amount']

    values = np.array([values[feature] for feature in column_order], dtype=object)

    if debug:
        print('Ordered feature values:')
        print(list(zip(column_order, values)))

    pred = model.predict(values.reshape(1, -1))
    return str(pred[0])

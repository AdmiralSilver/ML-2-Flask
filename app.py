from flask import Flask, render_template
import predict
from forms import DataForm

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
@app.route('/index', methods=['GET', 'POST'])
def index():

    form = DataForm()
    if form.validate_on_submit():
        data = {
            'budget': form.budget.data,
            'popularity': form.popularity.data,
            'runtime': form.runtime.data,
            'cast_amount': form.cast_amount.data,
            'crew_amount': form.crew_amount.data,
        }
        prediction = predict.predict(data)
        return render_template('index.html', form=form, prediction=prediction)


if __name__ == '__main__':
    app.run()

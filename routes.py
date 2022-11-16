from flask import render_template, request, session, redirect, url_for

from __init__ import app
from forms import DataForm
from predict import predict

app.config['SECRET_KEY'] = 'DAT158'


@app.route('/', methods=['GET', 'POST'])
@app.route('/index', methods=['GET', 'POST'])
def index():
    form = DataForm()
    if form.validate_on_submit():
        for fieldname, value in form.data.items():
            session[fieldname] = value

        user_info = request.headers.get('User-Agent')

        prediction = predict(session)

        session['user_info'] = user_info
        session['prediction'] = prediction
        print(session['prediction'])
        return redirect(url_for('index'))
    return render_template('index.html', form=form)


@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

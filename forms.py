from flask_wtf import FlaskForm
from wtforms import IntegerField, SubmitField
from wtforms.validators import DataRequired


class DataForm(FlaskForm):

    budget = IntegerField('Budget:?', validators=[DataRequired()])
    popularity = IntegerField('Popularity:?', validators=[DataRequired()])
    runtime = IntegerField('Runtime:?', validators=[DataRequired()])
    cast_amount = IntegerField('Cast Amount:?', validators=[DataRequired()])
    crew_amount = IntegerField('Crew Amount:?', validators=[DataRequired()])

    submit = SubmitField('Submit')

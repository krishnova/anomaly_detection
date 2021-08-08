from flask_wtf import FlaskForm
from wtforms import Form, StringField, BooleanField, SubmitField, FileField
from wtforms.validators import DataRequired, Email, EqualTo, ValidationError, Length, required


class upload_form(FlaskForm):
    uploadFile=FileField('.csv')
    column = StringField("Select Column", validators=[required()])
    submit = SubmitField("Submit")
from flask import Flask, Response, jsonify, request, render_template, redirect, url_for, flash
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask_wtf import FlaskForm
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('Agg')
import matplotlib
from psutil import users
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import DataRequired, Email, ValidationError
import pickle
import pandas as pd
import numpy as np
import logging
import bcrypt
import re
from flask_mysqldb import MySQL
import secrets
from datetime import datetime, timedelta
from flask_mail import Mail, Message
from wtforms.validators import DataRequired, Email, EqualTo
from flask import send_file
import csv
import io
from flask import send_file, make_response
import pdfkit
import shap 


app = Flask(__name__)
logging.basicConfig(level=logging.DEBUG)

# Configuration
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'loan_prediction_system'
app.config['SECRET_KEY'] = '5220'

mysql = MySQL(app)

app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = '1mrc7552@gmail.com'  
app.config['MAIL_PASSWORD'] = 'Road1021'   
app.config['MAIL_DEFAULT_SENDER'] = 'loanprediction@gmail.com'

mail = Mail(app)

# Flask-Login setup
login_manager = LoginManager(app)
login_manager.login_view = 'login'


class GaussianNaiveBayes:
    def fit(self, X, y):
        # Convert X to numeric values if it contains strings-
        X = X.apply(pd.to_numeric, errors='coerce')
        
        self.classes = np.unique(y)
        self.mean = np.zeros((len(self.classes), X.shape[1]))
        self.var = np.zeros((len(self.classes), X.shape[1]))
        self.priors = np.zeros(len(self.classes))
        
        for idx, c in enumerate(self.classes):
            X_c = X[y == c]
            self.mean[idx, :] = X_c.mean(axis=0)
            self.var[idx, :] = X_c.var(axis=0)
            self.priors[idx] = X_c.shape[0] / float(X.shape[0])

    def predict(self, X):
        # Convert X to numeric values if it contains strings
        X = X.apply(pd.to_numeric, errors='coerce')  # Add this line to handle potential strings in the prediction data
        y_pred = [self._predict(x) for x in X.values] # Convert X to a NumPy array
        return np.array(y_pred)

    def _predict(self, x):
        posteriors = []
        
        for idx, c in enumerate(self.classes):
            prior = np.log(self.priors[idx])
            posterior = np.sum(np.log(self._pdf(idx, x)))
            posterior = prior + posterior
            posteriors.append(posterior)
        
        return self.classes[np.argmax(posteriors)]

    def _pdf(self, class_idx, x):
        mean = self.mean[class_idx]
        var = self.var[class_idx]
        numerator = np.exp(- (x - mean) ** 2 / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator

    # New methods for probability prediction
    def predict_proba(self, X):
        X = X.apply(pd.to_numeric, errors='coerce')
        y_prob = [self._predict_proba(x) for x in X.values]
        return np.array(y_prob)

    def _predict_proba(self, x):
        posteriors = []
        for idx, c in enumerate(self.classes):
            prior = np.log(self.priors[idx])
            posterior = np.sum(np.log(self._pdf(idx, x)))
            posterior = prior + posterior
            posteriors.append(posterior)
        
        # Compute softmax probabilities
        exp_posteriors = np.exp(posteriors - np.max(posteriors))
        probabilities = exp_posteriors / np.sum(exp_posteriors)
        return probabilities

# Loading the trained model and feature names
with open('model.pkl', 'rb') as model_file:
    gnb, feature_names = pickle.load(model_file)

    
@app.route('/predict', methods=['POST'])
@login_required
def predict():
    try:
        all_features = feature_names
        input_data = {feature: 0 for feature in all_features}

        # Processing form data
        for feature in request.form:
            value = request.form[feature]
            if feature in ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History']:
                input_data[feature] = float(value) if value else 0
            else:
                categorical_feature = f"{feature}_{value}"
                if categorical_feature in all_features:
                    input_data[categorical_feature] = 1

        df = pd.DataFrame([input_data])

        # Model Prediction
        prediction = gnb.predict(df)
        predicted_probability = gnb.predict_proba(df)[0]
        
        result = 'Eligible' if prediction[0] == 1 else 'Not Eligible'
        probability = predicted_probability[1] if prediction[0] == 1 else predicted_probability[0]
        
        prediction_text = f'Loan Eligibility: {result}'
        predicted_value = f'{probability:.2f}'
        
        # Store prediction in database
        cursor = mysql.connection.cursor()
        query = """
        INSERT INTO loan_predictions 
        (user_id, applicant_income, coapplicant_income, loan_amount, loan_amount_term, credit_history, prediction, accuracy) 
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """
        cursor.execute(query, (
            current_user.id,
            float(request.form['ApplicantIncome']),
            float(request.form['CoapplicantIncome']),
            float(request.form['LoanAmount']),
            float(request.form['Loan_Amount_Term']),
            float(request.form['Credit_History']),
            result,
            float(probability)
        ))
        mysql.connection.commit()
        cursor.close()

        # Initialize SHAP contributions
        shap_contributions = {}

        # Generate SHAP values using KernelExplainer for Naive Bayes
        try:
            # Use mean values from the dataset as reference data
            reference_data = df.mean().values.reshape(1, -1)
            
            # Initialize KernelExplainer with Naive Bayes model
            explainer = shap.KernelExplainer(gnb.predict_proba, reference_data)
            
            # Compute SHAP values
            shap_values = explainer.shap_values(df)
            
            # Log SHAP values for debugging
            app.logger.info(f"SHAP values: {shap_values}")

            # Collect SHAP contributions for the 'Eligible' class (class 1)
            shap_contributions = {
                feature: shap_value for feature, shap_value in zip(df.columns, shap_values[1][0])
            }
        except Exception as shap_error:
            app.logger.error(f"SHAP error: {str(shap_error)}", exc_info=True)
            shap_contributions = {}  # Make sure it's an empty dictionary if error occurs

        # Generate graph for the prediction
        labels = ['Eligible', 'Not Eligible']
        probabilities = [predicted_probability[1], predicted_probability[0]]
        plt.bar(labels, probabilities, color=['green', 'red'])
        plt.title('Loan Eligibility Prediction')
        plt.xlabel('Outcome')
        plt.ylabel('Probability')

        # Save graph to a byte stream to be passed to HTML
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plt.close()

        # Make sure shap_contributions is always a dictionary when passed to the template
        return render_template('loan_predictor.html', 
                               prediction_text=prediction_text,
                               predicted_value=predicted_value, 
                               user=current_user, 
                               graph_url='/plot.png',
                               shap_contributions=shap_contributions if shap_contributions else None)
    except Exception as e:
        app.logger.error(f"Error in prediction: {str(e)}", exc_info=True)
        return render_template('loan_predictor.html', 
                               prediction_text=f'Error: {str(e)}', 
                               user=current_user)

# Route to serve the prediction graph image
@app.route('/plot.png')
def plot_png():
    img = io.BytesIO()
    plt.bar(['Eligible', 'Not Eligible'], [0.8, 0.2], color=['green', 'red'])  # Default/fallback values
    plt.savefig(img, format='png')
    img.seek(0)
    return Response(img.getvalue(), mimetype='image/png')
class User(UserMixin):
    def __init__(self, id, name, username,mobile_number, email):
        self.id = id
        self.name = name
        self.username = username
        self.mobile_number =mobile_number
        self.email = email
        self.is_admin = mobile_number is None
        self.is_admin = email is None 

    @staticmethod
    def get(user_id):
        cursor = mysql.connection.cursor()
        cursor.execute("SELECT * FROM user_info WHERE id = %s", (user_id,))
        user = cursor.fetchone()
        cursor.close()
        if not user:
            cursor = mysql.connection.cursor()
            cursor.execute("SELECT * FROM admins WHERE id = %s", (user_id,))
            admin = cursor.fetchone()
            cursor.close()
            if admin:
                return User(id=admin[0], name=admin[1], username=admin[2], mobile_number=None, email=None)
        else:
            return User(id=user[0], name=user[1], username=user[2],mobile_number=[4], email=user[5])
        return None
@login_manager.user_loader
def load_user(user_id):
    return User.get(user_id)

# Forms
class RegisterForm(FlaskForm):
    name = StringField("Name", validators=[DataRequired()])
    username = StringField("Username", validators=[DataRequired()])
    password = PasswordField("Password", validators=[DataRequired()])
    mobile_number = StringField("Mobile Number", validators=[DataRequired()])
    email = StringField("Email", validators=[DataRequired(), Email()])
    submit = SubmitField('Register')

    def validate_name(self, field):
        if not re.match("^[A-Za-z]+$", field.data):
            raise ValidationError("Name can only contain letters.")
        if len(field.data) > 30:
            raise ValidationError("Name must be 30 characters or fewer.")

    def validate_username(self, field):
        if not re.match("^[a-z0-9]+$", field.data):
            raise ValidationError("Username can only contain lowercase letters and numbers.")
        if len(field.data) > 15:
            raise ValidationError("Username must be 15 characters or fewer.")
        cursor = mysql.connection.cursor()
        cursor.execute("SELECT * FROM user_info where username = %s", (field.data,))
        user = cursor.fetchone()
        cursor.close()
        if user:
            raise ValidationError("Username Already Taken")

    def validate_email(self, field):
        cursor = mysql.connection.cursor()
        cursor.execute("SELECT * FROM user_info where email=%s ", (field.data,))
        user = cursor.fetchone()
        cursor.close()
        if user:
            raise ValidationError("Email Already Taken")

    def validate_mobile_number(self, field):
        mobile_number = field.data
        if len(mobile_number) != 10 or not (mobile_number.startswith('98') or mobile_number.startswith('97')):
            raise ValidationError("Mobile Number must be 10 digits long and start with 98 or 97.")

    def validate_password(self, field):
        password = field.data
        if len(password) > 30:
            raise ValidationError("Password must be less than 30 letters")

class LoginForm(FlaskForm):
    email = StringField("Email", validators=[DataRequired(), Email()])
    password = PasswordField("Password", validators=[DataRequired()])
    submit = SubmitField('Login')

class AdminRegisterForm(FlaskForm):
    name = StringField("Name", validators=[DataRequired()])
    username = StringField("Username", validators=[DataRequired()])
    password = PasswordField("Password", validators=[DataRequired()])
    submit = SubmitField('Register')

    def validate_name(self, field):
        if not re.match("^[A-Za-z]+$", field.data):
            raise ValidationError("Name can only contain letters.")
        if len(field.data) > 30:
            raise ValidationError("Name must be 30 characters or fewer.")

    def validate_username(self, field):
        if not re.match("^[a-z0-9]+$", field.data):
            raise ValidationError("Username can only contain lowercase letters and numbers.")
        if len(field.data) > 15:
            raise ValidationError("Username must be 15 characters or fewer.")
        cursor = mysql.connection.cursor()
        cursor.execute("SELECT * FROM admins where username = %s", (field.data,))
        user = cursor.fetchone()
        cursor.close()
        if user:
            raise ValidationError("Username Already Taken")

    
    def validate_password(self, field):
        password = field.data
        if len(password) > 30:
            raise ValidationError("Password must be less than 30 letters")
    
class AdminLoginForm(FlaskForm):
    username = StringField("username", validators=[DataRequired()])
    password = PasswordField("Password", validators=[DataRequired()])
    submit = SubmitField('Login')


# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    form = RegisterForm()
    if form.validate_on_submit():
        name = form.name.data
        username = form.username.data
        password = form.password.data
        hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
        mobile_number = form.mobile_number.data
        email = form.email.data
        
        cursor = mysql.connection.cursor()
        cursor.execute("INSERT INTO user_info(name,username,password,mobile_number,email) VALUES (%s,%s,%s,%s,%s)",
                       (name, username, hashed_password, mobile_number, email))
        mysql.connection.commit()
        cursor.close()

        return redirect(url_for('login'))

    return render_template('register.html', form=form)

@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        email = form.email.data
        password = form.password.data
        
        cursor = mysql.connection.cursor()
        cursor.execute("SELECT * FROM user_info WHERE email=%s", (email,))
        admin_data = cursor.fetchone()
        cursor.close()
        
        if admin_data and bcrypt.checkpw(password.encode('utf-8'), admin_data[3].encode('utf-8')):
            user = User(id=admin_data[0], name=admin_data[1], username=admin_data[2],mobile_number=[4], email=admin_data[5])
            login_user(user)
            return redirect(url_for('dashboard'))
        else:
            flash("Login failed. Please check your email and password")
            return redirect(url_for('login'))

    return render_template('login.html', form=form)

@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard.html', user=current_user)

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash("You have been logged out successfully")
    return redirect(url_for('login'))

@app.route('/loan_predictor')
@login_required
def home():
    return render_template('loan_predictor.html', user=current_user)

@app.route('/admin_register', methods=['GET', 'POST'])
def admin_register():
    form = AdminRegisterForm()
    if form.validate_on_submit():
        name = form.name.data
        username = form.username.data
        password = form.password.data
        hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
        
        cursor = mysql.connection.cursor()
        cursor.execute("INSERT INTO admins(name,username,password) VALUES (%s,%s,%s)",
                       (name, username, hashed_password))
        mysql.connection.commit()
        cursor.close()

        return redirect(url_for('admin_login'))

    return render_template('admin_register.html', form=form)


@app.route('/admin_login', methods=['GET', 'POST'])
def admin_login():
    form = AdminLoginForm()
    if form.validate_on_submit():
        username = form.username.data
        password = form.password.data

        cursor = mysql.connection.cursor()
        cursor.execute("SELECT * FROM admins WHERE username=%s", (username,))
        admin_data = cursor.fetchone()
        cursor.close()
        
        if admin_data and bcrypt.checkpw(password.encode('utf-8'), admin_data[3].encode('utf-8')):
            admin = User(id=admin_data[0], name=admin_data[1], username=admin_data[2],mobile_number=None, email=None)
            login_user(admin)
            return redirect(url_for('admin_dashboard'))
        else:
            flash("Login failed. Please check your username and password")
            return redirect(url_for('admin_login'))
    
    return render_template('admin_login.html', form=form)

@app.route('/admin_logout')
@login_required
def admin_logout():
    logout_user()
    flash("You have been logged out successfully")
    return redirect(url_for('admin_login'))

@app.route('/admin_dashboard')
@login_required
def admin_dashboard():
    if not current_user.is_admin:
        flash("You don't have permission to access the admin dashboard.")
        return redirect(url_for('admin_login'))
    
    cursor = mysql.connection.cursor()
    cursor.execute("SELECT COUNT(*) FROM user_info")
    total_users = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM loan_predictions")
    total_predictions = cursor.fetchone()[0]
    cursor.close()
    
    return render_template('admin_dashboard.html', user=current_user, total_users=total_users, total_predictions=total_predictions)

@app.route('/user_list')
@login_required
def user_list():
    if not current_user.is_admin:
        flash("You don't have permission to access this page.")
        return redirect(url_for('login'))
    
    cursor = mysql.connection.cursor()
    cursor.execute("SELECT id, name, username, mobile_number, email FROM user_info")
    users = cursor.fetchall()
    cursor.close()
    
    return render_template('user_list.html', user=current_user, users=users)

@app.route('/prediction_history')
@login_required
def prediction_history():
    if not current_user.is_admin:
        flash("You don't have permission to access this page.")
        return redirect(url_for('login'))
    
    cursor = mysql.connection.cursor()
    cursor.execute("""
        SELECT lp.*, ui.name 
        FROM loan_predictions lp 
        JOIN user_info ui ON lp.user_id = ui.id 
        ORDER BY lp.created_at DESC
    """)
    predictions = cursor.fetchall()
    cursor.close()
    
    return render_template('prediction_history.html', user=current_user, predictions=predictions)

@app.route('/download_users/<format>')
@login_required
def download_users(format):
    if not current_user.is_admin:
        flash("You don't have permission to access this page.")
        return redirect(url_for('admin_login'))
    
    cursor = mysql.connection.cursor()
    cursor.execute("SELECT id, name, username, mobile_number, email FROM user_info")
    users = cursor.fetchall()
    cursor.close()
    
    if format == 'csv':
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(['ID', 'Name', 'Username', 'Mobile Number', 'Email'])
        for user in users:
            writer.writerow(user)
        output.seek(0)
        return send_file(io.BytesIO(output.getvalue().encode('utf-8')),
                         mimetype='text/csv',
                         download_name='users.csv',
                         as_attachment=True)
    elif format == 'pdf':
        # For PDF, we'll use pdfkit as suggested earlier
        path_wkhtmltopdf = r'C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe'
        config = pdfkit.configuration(wkhtmltopdf=path_wkhtmltopdf)

        html = render_template('users_pdf.html', users=users)
        pdf = pdfkit.from_string(html, False, configuration=config)

        response = make_response(pdf)
        response.headers["Content-Type"] = "application/pdf"
        response.headers["Content-Disposition"] = "attachment; filename=users.pdf"
    return response

@app.route('/download_predictions/<format>')
@login_required
def download_predictions(format):
    if not current_user.is_admin:
        flash("You don't have permission to access this page.")
        return redirect(url_for('login'))
    
    cursor = mysql.connection.cursor()
    cursor.execute("""
        SELECT lp.*, ui.name 
        FROM loan_predictions lp 
        JOIN user_info ui ON lp.user_id = ui.id 
        ORDER BY lp.created_at DESC
    """)
    predictions = cursor.fetchall()
    cursor.close()
    
    if format == 'csv':
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(['ID', 'User ID', 'User Name', 'Applicant Income', 'Coapplicant Income', 'Loan Amount', 'Loan Amount Term', 'Credit History', 'Prediction', 'Accuracy', 'Created At'])
        for pred in predictions:
            writer.writerow(pred)
        output.seek(0)
        return send_file(io.BytesIO(output.getvalue().encode('utf-8')),
                         mimetype='text/csv',
                         download_name='predictions.csv',
                         as_attachment=True)
    elif format == 'pdf':
        path_wkhtmltopdf = r'C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe'
        config = pdfkit.configuration(wkhtmltopdf=path_wkhtmltopdf)
        html = render_template('predictions_pdf.html', predictions=predictions)
        pdf = pdfkit.from_string(html, False, configuration=config)
        response = make_response(pdf)
        response.headers["Content-Type"] = "application/pdf"
        response.headers["Content-Disposition"] = "attachment; filename=predictions.pdf"

    return response


@app.route('/user_history')
@login_required
def user_history():
    cursor = mysql.connection.cursor()
    cursor.execute("SELECT * FROM loan_predictions WHERE user_id = %s ORDER BY created_at DESC", (current_user.id,))
    history = cursor.fetchall()
    cursor.close()
    return render_template('user_history.html', history=history)

@app.route('/delete_prediction/<int:prediction_id>', methods=['POST'])
@login_required
def delete_prediction(prediction_id):
    cursor = mysql.connection.cursor()
    cursor.execute("DELETE FROM loan_predictions WHERE id = %s AND user_id = %s", (prediction_id, current_user.id))
    mysql.connection.commit()
    cursor.close()
    flash('Prediction deleted successfully', 'success')
    return redirect(url_for('user_history'))

@app.route('/delete_adminprediction/<int:prediction_id>', methods=['POST'])
@login_required
def delete_adminprediction(prediction_id):
    if not current_user.is_admin:
        flash("You don't have permission to access this page.")
        return redirect(url_for('login'))
    
    cursor = mysql.connection.cursor()
    cursor.execute("DELETE FROM loan_predictions WHERE id = %s", (prediction_id,))
    mysql.connection.commit()
    cursor.close()
    return jsonify({'success': True})

@app.route('/delete_user/<int:user_id>', methods=['POST'])
@login_required
def delete_user(user_id):
    if not current_user.is_admin:
        flash("You don't have permission to access this page.")
        return redirect(url_for('login'))
    
    cursor = mysql.connection.cursor()
    cursor.execute("DELETE FROM user_info WHERE id = %s", (user_id,))
    mysql.connection.commit()
    cursor.close()
    return jsonify({'success': True})


@app.route('/update_prediction/<int:prediction_id>', methods=['GET', 'POST'])
@login_required
def update_prediction(prediction_id):
    cursor = mysql.connection.cursor()
    if request.method == 'POST':
        # Gather updated data
        applicant_income = float(request.form['ApplicantIncome'])
        coapplicant_income = float(request.form['CoapplicantIncome'])
        loan_amount = float(request.form['LoanAmount'])
        loan_amount_term = float(request.form['Loan_Amount_Term'])
        credit_history = float(request.form['Credit_History'])

        # Prepare data for prediction
        input_data = {feature: 0 for feature in feature_names}
        input_data.update({
            'ApplicantIncome': applicant_income,
            'CoapplicantIncome': coapplicant_income,
            'LoanAmount': loan_amount,
            'Loan_Amount_Term': loan_amount_term,
            'Credit_History': credit_history
        })

        # Make prediction
        df = pd.DataFrame([input_data])
        prediction = gnb.predict(df)
        predicted_probability = gnb.predict_proba(df)[0]

        result = 'Eligible' if prediction[0] == 1 else 'Not Eligible'
        probability = predicted_probability[1] if prediction[0] == 1 else predicted_probability[0]

        # Update the prediction in the database
        query = """
        UPDATE loan_predictions 
        SET applicant_income = %s, coapplicant_income = %s, loan_amount = %s, 
        loan_amount_term = %s, credit_history = %s, prediction = %s, accuracy = %s
        WHERE id = %s AND user_id = %s
        """
        cursor.execute(query, (
            applicant_income,
            coapplicant_income,
            loan_amount,
            loan_amount_term,
            credit_history,
            result,
            float(probability),
            prediction_id,
            current_user.id
        ))
        mysql.connection.commit()
        flash('Prediction updated successfully', 'success')
        return redirect(url_for('user_history'))
    else:
        # Fetch the prediction data
        cursor.execute("SELECT * FROM loan_predictions WHERE id = %s AND user_id = %s", (prediction_id, current_user.id))
        prediction = cursor.fetchone()
        cursor.close()
        if prediction:
            return render_template('update_prediction.html', prediction=prediction)
        else:
            flash('Prediction not found', 'error')
            return redirect(url_for('user_history'))

class UserUpdateForm(FlaskForm):
    name = StringField("Name", validators=[DataRequired()])
    username = StringField("Username", validators=[DataRequired()])
    mobile_number = StringField("Mobile Number", validators=[DataRequired()])
    email = StringField("Email", validators=[DataRequired(), Email()])
    submit = SubmitField('Update')

    def __init__(self, original_username, original_email, *args, **kwargs):
        super(UserUpdateForm, self).__init__(*args, **kwargs)
        self.original_username = original_username
        self.original_email = original_email

    def validate_name(self, field):
        if not re.match("^[A-Za-z]+$", field.data):
            raise ValidationError("Name can only contain letters.")
        if len(field.data) > 30:
            raise ValidationError("Name must be 30 characters or fewer.")

    def validate_username(self, field):
        if field.data != self.original_username:
            if not re.match("^[a-z0-9]+$", field.data):
                raise ValidationError("Username can only contain lowercase letters and numbers.")
            if len(field.data) > 15:
                raise ValidationError("Username must be 15 characters or fewer.")
            cursor = mysql.connection.cursor()
            cursor.execute("SELECT * FROM user_info where username = %s", (field.data,))
            user = cursor.fetchone()
            cursor.close()
            if user:
                raise ValidationError("Username Already Taken")

    def validate_email(self, field):
        if field.data != self.original_email:
            cursor = mysql.connection.cursor()
            cursor.execute("SELECT * FROM user_info where email=%s ", (field.data,))
            user = cursor.fetchone()
            cursor.close()
            if user:
                raise ValidationError("Email Already Taken")

    def validate_mobile_number(self, field):
        mobile_number = field.data
        if len(mobile_number) != 10 or not (mobile_number.startswith('98') or mobile_number.startswith('97')):
            raise ValidationError("Mobile Number must be 10 digits long and start with 98 or 97.")

@app.route('/user_update/<int:user_id>', methods=['GET', 'POST'])
@login_required
def user_update(user_id):
    if not current_user.is_admin:
        flash("You don't have permission to access this page.")
        return redirect(url_for('login'))
    
    cursor = mysql.connection.cursor()
    cursor.execute("SELECT * FROM user_info WHERE id = %s", (user_id,))
    user = cursor.fetchone()
    cursor.close()
    
    if user is None:
        flash('User not found.', 'error')
        return redirect(url_for('user_list'))
    
    form = UserUpdateForm(original_username=user[2], original_email=user[5])
    
    if form.validate_on_submit():
        cursor = mysql.connection.cursor()
        try:
            cursor.execute("""
                UPDATE user_info
                SET name = %s, username = %s, mobile_number = %s, email = %s
                WHERE id = %s
            """, (form.name.data, form.username.data, form.mobile_number.data, form.email.data, user_id))
            mysql.connection.commit()
            flash('User updated successfully', 'success')
            return redirect(url_for('user_list'))
        except Exception as e:
            mysql.connection.rollback()
            flash(f'Error updating user: {str(e)}', 'error')
        finally:
            cursor.close()
    elif request.method == 'GET':
        form.name.data = user[1]
        form.username.data = user[2]
        form.mobile_number.data = user[4]
        form.email.data = user[5]
    
    return render_template('user_update.html', form=form, user_id=user_id)
class ForgotPasswordForm(FlaskForm):
    email = StringField('Email', validators=[DataRequired(), Email()])
    submit = SubmitField('Reset Password')

class ResetPasswordForm(FlaskForm):
    password = PasswordField('New Password', validators=[DataRequired()])
    confirm_password = PasswordField('Confirm New Password', validators=[DataRequired(), EqualTo('password')]) 
    submit = SubmitField('Change Password')

@app.route('/forgot_password', methods=['GET', 'POST'])
def forgot_password():
    form = ForgotPasswordForm()
    if form.validate_on_submit():
        email = form.email.data
        cursor = mysql.connection.cursor()
        cursor.execute("SELECT * FROM user_info WHERE email = %s", (email,))
        user = cursor.fetchone()
        if user:
            token = secrets.token_urlsafe(32)
            expiration = datetime.now() + timedelta(hours=1)
            cursor.execute("UPDATE user_info SET reset_token = %s, reset_token_expiration = %s WHERE email = %s",
                           (token, expiration, email))
            mysql.connection.commit()
            
            reset_url = url_for('reset_password', token=token, _external=True)
            msg = Message('Password Reset Request',
                          recipients=[email],
                          body=f'To reset your password, visit the following link: {reset_url}')
            mail.send(msg)
            
            flash('An email has been sent with instructions to reset your password.', 'info')
            return redirect(url_for('login'))
        else:
            flash('Email address not found.', 'error')
    return render_template('forgot_password.html', form=form)

@app.route('/reset_password/<token>', methods=['GET', 'POST'])
def reset_password(token):
    cursor = mysql.connection.cursor()
    cursor.execute("SELECT * FROM user_info WHERE reset_token = %s AND reset_token_expiration > %s",
                   (token, datetime.now()))
    user = cursor.fetchone()
    if user is None:
        flash('Invalid or expired reset token.', 'error')
        return redirect(url_for('login'))
    
    form = ResetPasswordForm()
    if form.validate_on_submit():
        hashed_password = bcrypt.hashpw(form.password.data.encode('utf-8'), bcrypt.gensalt())
        cursor.execute("UPDATE user_info SET password = %s, reset_token = NULL, reset_token_expiration = NULL WHERE id = %s",
                       (hashed_password, user[0]))
        mysql.connection.commit()
        flash('Your password has been updated.', 'success')
        return redirect(url_for('login'))
    return render_template('reset_password.html', form=form)

@app.route('/change_password', methods=['GET', 'POST'])
@login_required
def change_password():
    form = ResetPasswordForm()
    if form.validate_on_submit():
        hashed_password = bcrypt.hashpw(form.password.data.encode('utf-8'), bcrypt.gensalt())
        cursor = mysql.connection.cursor()
        cursor.execute("UPDATE user_info SET password = %s WHERE id = %s",
                       (hashed_password, current_user.id))
        mysql.connection.commit()
        flash('Your password has been updated.', 'success')
        return redirect(url_for('dashboard'))
    return render_template('change_password.html', form=form)

if __name__ == "__main__":
    app.run(debug=True)
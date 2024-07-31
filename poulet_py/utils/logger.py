import csv
from datetime import datetime
import os
import pandas as pd
import os
import ast
import platform
import tkinter as tk
from tkinter import messagebox
import sys
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from PIL import Image, ImageTk

def printme(message):
    print(f"\n{message}\n")


class SessionLogger:
    """
    A class to log experimental sessions by collecting various data points from the user
    and writing them to a logbook.
    """

    def __init__(self, path):
        self.file_names = [
            "logbook.csv",
            "subjects.csv",
            "licenses.csv",
            "methods.csv",
            "experimenters.csv",
            "experimental_designs.csv",
            "genotypes.csv",
            "drugs.csv",
            "todos.csv",
            "ear_marks.png",
        ]
        self.path = path
        self.paths = {
            os.path.splitext(file_name)[0]: os.path.join(path, file_name)
            for file_name in self.file_names
        }

        self.subject_id = None
        self.subject_ids = []
        self.license = None
        self.subproject = None
        self.method = None
        self.method_version = None
        self.duration_s = None
        self.condition = None
        self.experimenter = None
        self.todo = None
        self.emails = None
        self.email_mfa_password = None
        self.notes = None

        self.body_weight_trends_C57BL_6J = {
        'male': {
            'week_3': {'low_bound': 8.7, 'high_bound': 12.5},
            'week_4': {'low_bound': 13.9, 'high_bound': 19.1},
            'week_5': {'low_bound': 18.9, 'high_bound': 22.5},
            'week_6': {'low_bound': 20.1, 'high_bound': 23.7},
            'week_7': {'low_bound': 22.1, 'high_bound': 25.1},
            'week_8': {'low_bound': 23.6, 'high_bound': 26.4},
            'week_9': {'low_bound': 24.5, 'high_bound': 27.7},
            'week_10': {'low_bound': 25.2, 'high_bound': 28.6},
            'week_11': {'low_bound': 25.8, 'high_bound': 29.6},
            'week_12': {'low_bound': 26.9, 'high_bound': 30.9},
            'week_13': {'low_bound': 27.9, 'high_bound': 32.1},
            'week_14': {'low_bound': 28.6, 'high_bound': 33.0},
            'week_15': {'low_bound': 29.2, 'high_bound': 34.0},
            'week_16': {'low_bound': 29.7, 'high_bound': 34.5},
            'week_17': {'low_bound': 30.2, 'high_bound': 35.4},
            'week_18': {'low_bound': 30.5, 'high_bound': 36.1},
            'week_19': {'low_bound': 30.9, 'high_bound': 36.5},
            'week_20': {'low_bound': 31.3, 'high_bound': 37.1},
            'week_21': {'low_bound': 31.7, 'high_bound': 37.5},
            'week_22': {'low_bound': 31.9, 'high_bound': 38.3},
            'week_23': {'low_bound': 32.6, 'high_bound': 39.0},
            'week_24': {'low_bound': 32.9, 'high_bound': 39.7},
            },
        'female': {
            'week_3': {'low_bound': 8.4, 'high_bound': 11.8},
            'week_4': {'low_bound': 12.9, 'high_bound': 16.5},
            'week_5': {'low_bound': 16.7, 'high_bound': 18.9},
            'week_6': {'low_bound': 17.6, 'high_bound': 19.4},
            'week_7': {'low_bound': 18.0, 'high_bound': 20.0},
            'week_8': {'low_bound': 18.4, 'high_bound': 20.8},
            'week_9': {'low_bound': 19.0, 'high_bound': 21.6},
            'week_10': {'low_bound': 19.3, 'high_bound': 22.1},
            'week_11': {'low_bound': 19.8, 'high_bound': 22.8},
            'week_12': {'low_bound': 20.3, 'high_bound': 23.5},
            'week_13': {'low_bound': 20.7, 'high_bound': 24.5},
            'week_14': {'low_bound': 21.0, 'high_bound': 25.0},
            'week_15': {'low_bound': 21.2, 'high_bound': 25.8},
            'week_16': {'low_bound': 21.6, 'high_bound': 26.2},
            'week_17': {'low_bound': 21.6, 'high_bound': 26.6},
            'week_18': {'low_bound': 21.9, 'high_bound': 27.1},
            'week_19': {'low_bound': 22.0, 'high_bound': 27.6},
            'week_20': {'low_bound': 22.5, 'high_bound': 28.1},
            'week_21': {'low_bound': 22.6, 'high_bound': 29.0},
            'week_22': {'low_bound': 22.9, 'high_bound': 29.3},
            'week_23': {'low_bound': 23.2, 'high_bound': 29.8},
            'week_24': {'low_bound': 23.5, 'high_bound': 30.3},
            },
        }

        self.clear_input_buffer()

    def set_email_details(self, emails, email_mfa_password):
        """
        Set the email details for sending emails.
        """
        self.emails = emails
        self.email_mfa_password = email_mfa_password

    def get_subject_id(self):
        """
        Prompts user to select multiple subject IDs and retrieves the corresponding subject numbers.
        """
        subjects_data_dict = self.get_csv_data(self.paths["subjects"])
        subjects_data_dict = {k: v for k, v in subjects_data_dict.items() if v["active"]}

        root = tk.Tk()
        root.title("Select Subject IDs")

        def toggle_selection(subject_id, button):
            if subject_id in self.subject_ids:
                self.subject_ids.remove(subject_id)
                button.config(relief="raised", bg="lightyellow")
            else:
                self.subject_ids.append(subject_id)
                button.config(relief="sunken", bg="lightgreen")

        def submit():
            result = ', '.join(self.subject_ids)
            print(f"Selected Subject IDs: {result}")
            root.destroy()

        def on_closing():
            if messagebox.askokcancel("Quit", "Do you want to quit?"):
                root.destroy()
                sys.exit()

        root.protocol("WM_DELETE_WINDOW", on_closing)

        frame = tk.Frame(root)
        frame.pack(side=tk.LEFT, padx=10, pady=10)

        max_columns = 3  # Set the maximum number of columns
        row = 0
        column = 0

        def create_button(subject_id, row, column):
            button = tk.Button(frame, text=subject_id, width=20,
                               command=lambda: toggle_selection(subject_id, button),
                               bg="lightyellow")
            button.grid(row=row, column=column, padx=5, pady=5)
            return button

        for subject_id, details in subjects_data_dict.items():
            create_button(subject_id, row, column)
            column += 1
            if column >= max_columns:
                column = 0
                row += 1

        tk.Button(frame, text="Accept", command=submit).grid(row=row+1, column=0, columnspan=max_columns, pady=10)

        # Load and display the image
        image_frame = tk.Frame(root)
        image_frame.pack(side=tk.RIGHT, padx=10, pady=10)

        img = Image.open(self.paths["ear_marks"])
        max_size = (600, 600)
        img.thumbnail(max_size, Image.Resampling.LANCZOS)

        photo = ImageTk.PhotoImage(img)

        label = tk.Label(image_frame, image=photo)
        label.image = photo  # Keep a reference to avoid garbage collection
        label.pack()

        root.mainloop()

    def get_license_data(self):
        """
        Retrieves the license data for the subject.
        """

        self.license = self.get_current_license()
        if self.license in ["ZH_139", "X9016_21", "G0167_23"]:
            license_data = self.get_csv_data(self.paths["licenses"])
            self.license = self.get_input(
                "Enter the license", list(license_data.keys())
            )
            # update the current license in the subjects.csv file
            subjects_data_csv = pd.read_csv(self.paths["subjects"])
            subjects_data_csv.loc[
                subjects_data_csv["subject_id"] == self.subject_id, "current_license"
            ] = self.license
            #get the genotype of the mouse
            genotype = subjects_data_csv.loc[
                subjects_data_csv["subject_id"] == self.subject_id, "genotype"
            ].iloc[0]
            # get the cage number of the mouse
            cage_number = subjects_data_csv.loc[
                subjects_data_csv["subject_id"] == self.subject_id, "cage_number"
            ].iloc[0]

            subjects_data_csv.to_csv(self.paths["subjects"], index=False)

            if self.emails is not None and self.email_mfa_password is not None:
                subject = 'Mouse change of license'
                #get the date as dd/mm/yyyy
                date = datetime.now().strftime("%d/%m/%Y")
                self.send_email(subject, f"On {date}, the mouse {self.subject_id} with genotype {genotype} has been assigned the license {self.license}. This mouse is in cage {cage_number}.\n\n- Date: {date}\n- License: {self.license}\n- Genotype: {genotype}\n- Cage number: {cage_number}\n\nThanks!")

        printme(f"License: {self.license}")

    @staticmethod
    def calculate_age(date_of_birth):
        """
        Calculate the age of the mouse based on its birth date.

        Args:
            birth_date (str): The birth date of the mouse in the format YYYY-MM-DD.

        Returns:
            int: The age of the mouse in days.
        """
        birth_date = datetime.strptime(date_of_birth, '%d/%m/%Y')
        current_date = datetime.now()
        age_in_days = (current_date - birth_date).days
        age_in_weeks = age_in_days / 7
        return int(age_in_weeks) if age_in_weeks - int(age_in_weeks) < 0.5 else int(age_in_weeks) + 1

    def send_email(self, subject, body, smtp_user = 'ivan.eromano@gmail.com'):
        """
            Send an email with the specified subject and body to the list of recipients.

            Args:
                subject (str): Subject of the email.
                body (str): Body of the email.
                to_emails (list): List of recipient email addresses.
        """
        # SMTP server configuration for Gmail
        smtp_server = "smtp.gmail.com"
        smtp_port = 587  # TLS
        smtp_password = self.email_mfa_password

        # Create the email
        msg = MIMEMultipart()
        msg['From'] = smtp_user
        msg['To'] = ", ".join(self.emails)
        msg['Subject'] = subject

        # Attach the email body
        msg.attach(MIMEText(body, 'plain'))

        try:
            # Connect to the SMTP server
            server = smtplib.SMTP(smtp_server, smtp_port)
            server.starttls()  # Upgrade the connection to a secure encrypted SSL/TLS connection
            server.login(smtp_user, smtp_password)
            
            # Send the email
            server.sendmail(smtp_user, self.emails, msg.as_string())
            print(f"Email sent to {', '.join(self.emails)}")
        except Exception as e:
            print(f"Failed to send email. Error: {e}")
        finally:
            # Close the connection
            server.quit()

    def get_subproject_data(self):
        """
        Retrieves the subproject data for the subject.
        """

        self.subproject = self.get_current_subproject()
        if self.subproject is None:
            license_data = self.get_csv_data(self.paths["licenses"])
            subprojects = eval(license_data[self.license]["subprojects"])
            
            if len(subprojects) == 1:
                self.subproject = subprojects[0]
            elif len(subprojects) == 0:
                self.subproject = ""
            else:
                self.subproject = self.get_input("Enter the subproject", subprojects)
                
            # update the current subproject in the subjects.csv file
            subjects_data_csv = pd.read_csv(self.paths["subjects"])
            subjects_data_csv.loc[
                subjects_data_csv["subject_id"] == self.subject_id, "current_subproject"
            ] = self.subproject
            subjects_data_csv.to_csv(self.paths["subjects"], index=False)

        printme(f"Subproject: {self.subproject}")

    def get_method_data(self):
        """
        Prompts user to select method and returns it.
        """
        if self.method is None:
            method_data_dict = self.get_csv_data(self.paths["methods"])
            root = tk.Tk()

            selected_method = tk.StringVar()

            def select_method(method, button):
                selected_method.set(method)
                for btn in buttons:
                    btn.config(relief="raised", bg="lightyellow")
                button.config(relief="sunken", bg="lightgreen")

            def submit():
                self.method = selected_method.get()
                root.destroy()

            def on_closing():
                if messagebox.askokcancel("Quit", "Do you want to quit?"):
                    root.destroy()
                    sys.exit()

            root.protocol("WM_DELETE_WINDOW", on_closing)

            frame = tk.Frame(root)
            frame.pack(padx=10, pady=10)

            buttons = []
            row = 0
            for method in sorted(method_data_dict.keys()):
                def create_button(method, row):
                    button = tk.Button(frame, text=method, width=20,
                                       command=lambda: select_method(method, button),
                                       bg="lightyellow")
                    button.grid(row=row, column=0, padx=5, pady=5)
                    return button
                
                buttons.append(create_button(method, row))
                row += 1

            tk.Button(frame, text="Accept", command=submit).grid(row=row, column=0, pady=10)

            root.mainloop()

        # check whether the method requires drugs
        methods_data_csv = pd.read_csv(self.paths["methods"])
        self.drugs_required = methods_data_csv.loc[
            methods_data_csv["name"] == self.method, "drugs"
        ].iloc[0]
        # check whether the method means logging out
        self.logged_out = methods_data_csv.loc[
            methods_data_csv["name"] == self.method, "logging_out"
        ].iloc[0]
        # check whether the method requires a todo
        self.todo = methods_data_csv.loc[
            methods_data_csv["name"] == self.method, "todo"
        ].iloc[0]
        # if the method requires a todo, get the todo_message
        if self.todo:
            self.todo_message = methods_data_csv.loc[
                methods_data_csv["name"] == self.method, "todo_message"
            ].iloc[0]
            self.todo_deadline = methods_data_csv.loc[
                methods_data_csv["name"] == self.method, "todo_deadline_h"
            ].iloc[0]

        print(f"Method: {self.method} ({'drugs required' if self.drugs_required else 'no drugs required'}) ({'logged out' if self.logged_out else 'not logged out'})")

    def get_method_version_data(self):
        """
        Prompts user to select method version and returns it.
        """
        if self.method_version is None:
            method_data = self.get_csv_data(self.paths["methods"])
            method_versions = eval(method_data[self.method]["versions"])
            self.method_version = self.get_input("Enter the version", method_versions)

        printme(f"Method version: {self.method_version}")

    def get_experimenter_data(self):
        """
        Prompts user to select experimenter and returns it.
        """
        if self.experimenter is None:
            experimenter_data = self.get_csv_data(self.paths["experimenters"])
            self.experimenter = self.get_input(
                "Enter the experimenter", list(experimenter_data.keys())
            )

        printme(f"Experimenter: {self.experimenter}")

    def get_condition_data(self):
        """
        Retrieves the condition data for the subject.
        """
        self.condition = self.get_mouse_condition()
        if self.condition is None:
            conditions_data_csv = pd.read_csv(self.paths["experimental_designs"])
            # get the rows with license_numner is self.license
            license_rows = conditions_data_csv[
                conditions_data_csv["license_number"] == self.license
            ]
            # get the rows with subproject is self.subproject
            subproject_rows = license_rows[
                license_rows["subproject"] == self.subproject
            ]
            # get the values in the column condition
            condition_data = subproject_rows["condition"].unique().tolist()

            # print(condition_data)
            self.condition = self.get_input("Enter the condition", list(condition_data))

            # update the mouse in the experimental_designs.csv file
            # find the row in which the self.condition is in the column condition
            condition_row = conditions_data_csv[
                (conditions_data_csv["condition"] == self.condition)
            ]

            # get the subjects column from the row
            subjects = condition_row["subjects"].apply(
                ast.literal_eval
            )

            # add the new subject_id to the subjects column
            subjects = subjects.iloc[0]
            subjects.append(self.subject_id)

            # update the subjects column in the row
            conditions_data_csv.loc[
                conditions_data_csv["condition"] == self.condition, "subjects"
            ] = str(subjects)

            # write the updated data to the CSV
            conditions_data_csv.to_csv(self.paths["experimental_designs"], index=False)

        printme(f"Condition: {self.condition}")

    def get_duration_data(self):
        """
        Prompts user to enter duration of the experiment and select a time unit, then returns it.
        """
        if self.duration_s is None:
            def convert_to_seconds(value, unit):
                if unit == "seconds":
                    return value
                elif unit == "minutes":
                    return value * 60
                elif unit == "hours":
                    return value * 3600
                elif unit == "days":
                    return value * 86400

            root = tk.Tk()
            root.title("Enter Duration of the Experiment")

            duration_var = tk.StringVar()
            unit_var = tk.StringVar(value="seconds")

            tk.Label(root, text="Enter duration:").grid(row=0, column=0, padx=5, pady=5)
            tk.Entry(root, textvariable=duration_var).grid(row=0, column=1, padx=5, pady=5)

            tk.Label(root, text="Select unit:").grid(row=1, column=0, padx=5, pady=5)
            
            unit_options = [("seconds", "seconds"), ("minutes", "minutes"), ("hours", "hours"), ("days", "days")]
            row = 1
            for text, value in unit_options:
                row += 1
                tk.Radiobutton(root, text=text, variable=unit_var, value=value).grid(row=row, column=1, sticky="w")

            def submit():
                try:
                    duration_value = int(duration_var.get())
                    duration_unit = unit_var.get()
                    self.duration_s = convert_to_seconds(duration_value, duration_unit)
                    
                    root.destroy()
                except ValueError:
                    messagebox.showerror("Invalid Input", "Please enter a valid number for duration.")

            def on_closing():
                if messagebox.askokcancel("Quit", "Do you want to quit?"):
                    root.destroy()
                    sys.exit()

            root.protocol("WM_DELETE_WINDOW", on_closing)

            tk.Button(root, text="Accept", command=submit).grid(row=row + 1, column=0, columnspan=2, pady=10)

            root.mainloop()
        else:
            printme(f"Duration defined in script")

        printme(f"Duration: {self.duration_s} seconds")

    def get_notes_data(self):
        """
        Prompts user to enter additional notes and returns it.
        """
        if self.notes is None:
            self.notes = input("Enter additional notes: ")

        printme(f"Notes: {self.notes}")

    def define_session(self):
        """
        Logs a session by collecting various data points from the user and writing them to a logbook.
        """
        self.clear_input_buffer()
        self.get_method_data()
        if self.method != "weighing":
            self.get_duration_data()
            self.get_method_version_data()

        self.get_experimenter_data()
        self.get_notes_data()

        if self.drugs_required:
            self.get_drugs_data()

        for self.subject_id in self.subject_ids:            
            if self.method == "weighing":
                self.log_weight()
            else:
                self.get_license_data()

                self.get_subproject_data()

                self.get_condition_data()

                if self.logged_out:
                    self.update_logged_out()

            self.log_session()
            self.set_todo()

    def set_todo(self):
        """
        Set a todo for the selected subject IDs.
        """
        if self.todo:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            # get the time and date now
            now = datetime.now()
            # get the date of the deadline
            if self.todo_deadline != "open" and self.todo_deadline is not None:
                #transform the striong to integers
                deadline = int(self.todo_deadline)

                # add the hours of the deadline to the current time
                notification_time = now + pd.Timedelta(hours=deadline)

                # if the notification time is not within 9am and 5pm, set it to 9am if it's after midnight and to 5pm if it's before midnight
                if notification_time.hour < 9:
                    notification_time = notification_time.replace(hour=9, minute=0, second=0)
                elif notification_time.hour >= 17:
                    notification_time = notification_time.replace(hour=15, minute=0, second=0)
                
                # format the notification time to the format YY-MM-DD HH:MM:SS
                self.todo_deadline_date = notification_time.strftime("%Y-%m-%d %H:%M:%S")

            # append to the todos.csv file timestamp,subject_id,deadline,message
            file_exists = os.path.isfile(self.paths["todos"])
            todo_entry = [timestamp, self.subject_id, self.todo_deadline_date, self.todo_message]

            with open(self.paths["todos"], mode="a", newline="") as file:
                writer = csv.writer(file)
                if not file_exists:
                    writer.writerow(
                        [
                            "timestamp",
                            "subject_id",
                            "deadline",
                            "message"
                        ]
                    )
                writer.writerow(todo_entry)

    def get_drugs_data(self):
        drugs_data_csv = pd.read_csv(self.paths["drugs"])
        
        # Create the main window
        root = tk.Tk()
        root.title("Drug Quantity Input")

        # Dictionary to hold the repeat values for each drug
        repeat_values = {row['name']: 0 for _, row in drugs_data_csv.iterrows()}
        labels = []  # List to hold references to label widgets

        def update_label(name, label):
            label.config(text=str(repeat_values[name]))

        def increment(name, label):
            repeat_values[name] += 1
            update_label(name, label)

        def decrement(name, label):
            if repeat_values[name] > 0:
                repeat_values[name] -= 1
                update_label(name, label)

        def submit():
            drugs_info = []
            for _, row in drugs_data_csv.iterrows():
                name = row['name']
                default_quantity = row['default_quantity']
                unit = row['unit']
                repeat = repeat_values[name]
                if repeat > 0:
                    drugs_info.append(f"{name}: {repeat} ({repeat * default_quantity} {unit})")

            result = '; '.join(drugs_info)
            print(result)
            root.destroy()  # Properly close the window and end the application

        # Create and place widgets for each drug
        for idx, (_, row) in enumerate(drugs_data_csv.iterrows()):
            name = row['name']
            default_quantity = row['default_quantity']
            unit = row['unit']

            tk.Label(root, text=f"{name} ({default_quantity} {unit})").grid(row=idx, column=0)
            tk.Button(root, text="-", command=lambda n=name, l=idx: decrement(n, labels[l])).grid(row=idx, column=1)
            label = tk.Label(root, text="0")
            label.grid(row=idx, column=2)
            labels.append(label)
            tk.Button(root, text="+", command=lambda n=name, l=idx: increment(n, labels[l])).grid(row=idx, column=3)

        tk.Button(root, text="Accept", command=submit).grid(row=len(drugs_data_csv), columnspan=4)

        root.mainloop()

    def update_logged_out(self):
        """
        Updates the logged_out field in the subjects.csv file.
        """
        subjects_data_csv = pd.read_csv(self.paths["subjects"])
        subjects_data_csv.loc[
            subjects_data_csv["subject_id"] == self.subject_id, "logged_out"
        ] = True
        #add logged out date
        subjects_data_csv.loc[
            subjects_data_csv["subject_id"] == self.subject_id, "logged_out_date"
        ] = datetime.now().strftime("%d/%m/%Y")
        subjects_data_csv.to_csv(self.paths["subjects"], index=False)

    def define_multiple_sessions(self):
        """
        Logs multiple sessions for selected subject IDs with common parameters.
        """
        self.clear_input_buffer()
        subject_ids = self.select_multiple_subjects()
        self.get_method_data()
        self.get_method_version_data()
        self.get_experimenter_data()
        self.get_duration_data()
        self.get_notes_data()

        for subject_id in subject_ids:
            self.subject_id = subject_id

            # License data
            self.get_license_data()

            # Subproject data
            self.get_subproject_data()

            # Condition data
            self.condition = self.get_mouse_condition()

            # Log the session
            self.log_session()

    @staticmethod
    def check_weight_bounds(sex, week, weight, body_weight_trends_C57BL_6J):
        """
        Check the weight of the subject against the body weight trends and show a warning if it's outside the bounds.

        Parameters:
        sex (str): The sex of the subject ('male' or 'female').
        week (int): The age of the subject in weeks.
        weight (float): The weight of the subject in grams.
        """
        week = min(week, 24)  # Limit the week value to 24
        week_key = f"week_{week}"
        trends = body_weight_trends_C57BL_6J[sex.lower()]

        if week_key in trends:
            low_bound = trends[week_key]['low_bound']
            high_bound = trends[week_key]['high_bound']

            if weight < low_bound or weight > high_bound:
                diff_low = weight - low_bound
                diff_high = high_bound - weight
                alert_message = (f"\n\nSubject of sex '{sex}' and age '{week}' weeks has a weight of '{weight}' grams.\n"
                                f"Expected weight bounds are {low_bound} to {high_bound} grams.\n"
                                f"Weight is {'below' if weight < low_bound else 'above'} the expected range by "
                                f"{abs(diff_low) if weight < low_bound else abs(diff_high)} grams.\n\n")
                show_alert(sex, alert_message)


    def log_weight(self):
        """
        Logs the weight of the subject by adding an entry to the weight logbook.
        """
        self.clear_input_buffer()
        subjects_data_csv = pd.read_csv(self.paths["subjects"])
        # Subject ID
        if self.subject_id is None:
            subjects_data_dict = self.get_csv_data(self.paths["subjects"])
            subjects_options = [f"{key}" for key, _ in subjects_data_dict.items()]
            subject_id = self.get_input(
                "Enter the ID of the subject", subjects_options, start=0
            )
            self.subject_id = subject_id.split()[0]

        print(f"Subject ID: {self.subject_id}")

        # Weight data
        weight = input("Enter the weight of the subject (in grams): ")
        self.weight = int(weight) if weight.isdigit() else None
        try:
            self.weight = float(weight)
        except ValueError:
            self.weight = None

        # get the date in the format DD/MM/YYYY
        date = datetime.now().strftime("%d/%m/%Y")

        # Add the weight entry to the CSV
        # check whether there's a row for the subject_id in the CSV
        if subjects_data_csv["subject_id"].isin([self.subject_id]).any():
            # Retrieve the current cell value
            current_value = subjects_data_csv.loc[
                subjects_data_csv["subject_id"] == self.subject_id, "weight"
            ].iloc[0]
            # get the age of the mouse
            self.age_in_weeks = self.calculate_age(subjects_data_csv.loc[
                subjects_data_csv["subject_id"] == self.subject_id, "date_of_birth"
            ].iloc[0])

            # get the sex
            sex = subjects_data_csv.loc[
                subjects_data_csv["subject_id"] == self.subject_id, "sex"
            ].iloc[0]

            self.check_weight_bounds(sex, self.age_in_weeks, self.weight, self.body_weight_trends_C57BL_6J)

            # Check if the cell is not empty and contains a dictionary
            if pd.notna(current_value):
                # Convert the string back to a dictionary
                current_dict = ast.literal_eval(current_value)
            else:
                # If the cell is empty, initialize a new dictionary
                current_dict = {}

            # Add the new date and weight
            current_dict[date] = self.weight

            # check whether there's more than one weight entry
            if len(current_dict) > 1:
                # get the last two weight entries
                # last_two_entries = list(current_dict.keys())[-2:]
                last_two_weights = list(current_dict.values())[-2:]
                # check whether the subject has lost more than 25% of its weight in the last two entries
                factor_weight_loss = (last_two_weights[1] - last_two_weights[0]) / last_two_weights[0]
                if  factor_weight_loss < -0.25:
                    alert_message = (f"\n\nSubject {self.subject_id} has lost more than 25% of its weight "
                                    f"in the last two dates.\n\n")
                    show_alert(self.subject_id, alert_message)
                else:
                    weight_diff = last_two_weights[1] - last_two_weights[0]
                    print(f"\n\nThe weight difference of subject {self.subject_id} is {weight_diff} grams.\n\n")


            # Convert the dictionary back to a string and update the DataFrame
            subjects_data_csv.loc[
                subjects_data_csv["subject_id"] == self.subject_id, "weight"
            ] = str(current_dict)

            # write the updated data to the CSV
            subjects_data_csv.to_csv(self.paths["subjects"], index=False)

            self.license = self.get_current_license()
            self.subproject = self.get_current_subproject()
            self.method_version = "101"
            self.duration_s = 60
            self.condition = self.get_mouse_condition()
            self.notes = f"Weight of {str(self.weight)} grams"

            print(
                f"Weight of {self.weight} grams logged for subject {self.subject_id} on {date}."
            )
        else:
            print(f"Subject ID {self.subject_id} not found.")

    def get_current_license(self):
        """
        Given a subject_id, output the corresponding license from the CSV file.

        Parameters:
            subject_id (str): ID of the subject.

        Returns:
            str: The license corresponding to the provided subject_id.
            None: If the subject_id is not found.
        """

        # Read the CSV file into a DataFrame
        df = pd.read_csv(self.paths["subjects"])
        # Search for the subject_id and get the corresponding license
        subject_row = df[df["subject_id"] == self.subject_id]

        if not subject_row.empty:
            return subject_row.iloc[0]["current_license"]
        else:
            return None

    def get_current_subproject(self):
        """
        Given a subject_id, output the corresponding subproject from the CSV file.

        Parameters:
            subject_id (str): ID of the subject.

        Returns:
            str: The subproject corresponding to the provided subject_id.
            None: If the subject_id is not found.
        """

        # Read the CSV file into a DataFrame
        df = pd.read_csv(self.paths["subjects"])
        # Search for the subject_id and get the corresponding subproject
        subject_row = df[df["subject_id"] == self.subject_id]

        if not pd.isnull(subject_row.iloc[0]["current_subproject"]):
            return subject_row.iloc[0]["current_subproject"]
        else:
            return None

    def get_mouse_condition(self):
        """
        Given a subject_id, output the corresponding condition from the CSV file.

        Returns:
            str: The condition corresponding to the provided subject_id.
            None: If the subject_id is not found.
        """

        if (
            self.license != "ZH_139"
            or self.license != "X9016/21"
        ):

            # Read the CSV file into a DataFrame
            df = pd.read_csv(self.paths["experimental_designs"])

            # Get the rows in which the license_number is self.license
            license_rows = df[df["license_number"] == self.license]

            # Get the rows in which the subproject is self.subproject
            subproject_rows = license_rows[
                license_rows["subproject"] == self.subproject
            ]

            # Convert the 'subjects' column from string to list
            subproject_rows["subjects"] = subproject_rows["subjects"].apply(
                ast.literal_eval
            )

            subject_row = subproject_rows[
                subproject_rows["subjects"].apply(lambda x: self.subject_id in x)
            ]


            if not subject_row.empty:
                return subject_row.iloc[0]["condition"]
            else:
                return None
        elif self.license == "ZH_139":
            return 'killing'
        elif self.license == "X9016/21":
            return 'teaching'

    def log_session(self):
        """
        Logs the session by adding an entry to the logbook.
        """

        if self.subject_id is None:
            raise ValueError("subject_id must be provided.")

        os.makedirs(os.path.dirname(self.paths["logbook"]), exist_ok=True)

        log_entry_data = [
            str(self.subject_id),
            str(self.license),
            str(self.subproject),
            str(self.method),
            str(self.method_version),
            str(self.duration_s),
            str(self.condition),
            str(self.experimenter),
            str(self.notes),
        ]
        log_entry_with_timestamp = self.append_timestamp(log_entry_data)

        try:
            file_exists = os.path.isfile(self.paths["logbook"])
            with open(self.paths["logbook"], mode="a", newline="") as file:
                writer = csv.writer(file)
                if not file_exists:
                    writer.writerow(
                        [
                            "timestamp",
                            "subject_id",
                            "license",
                            "subproject",
                            "method",
                            "method_version",
                            "duration_s",
                            "condition",
                            "experimenter",
                            "notes",
                        ]
                    )
                writer.writerow(log_entry_with_timestamp)
            print(f"Log entry added: {log_entry_with_timestamp}")

        except Exception as e:
            print(f"Error adding log entry: {e}")

    def add_subjects(self):
        """
        Adds new subjects to the subjects.csv file.
        """
        self.clear_input_buffer()
        subjects_data_csv = pd.read_csv(self.paths["subjects"])
        genotypes_data_csv = pd.read_csv(self.paths["genotypes"])

        new_subjects = []

        while True:

            # Add subject ID
            while True:
                subject = {}
                subject_id = input("Enter the subject ID: ")
                confirmation = input(
                    f"Confirm subject ID '{subject_id}' (y/n): "
                ).lower()
                if confirmation == "y":
                    # append the subject_id to the subject dictionary
                    subject["subject_id"] = subject_id
                    new_subjects.append(subject)
                    break
                elif confirmation == "n":
                    continue
                else:
                    printme("Invalid input. Please enter 'y' for yes or 'n' for no.")

            # Confirm adding more subjects
            add_more_ids = input(
                "Do you want to add more IDs for the same cage? (y/n): "
            ).lower()
            if add_more_ids == "n":
                break

        # Add sex
        sex = self.get_input("Enter the sex (male/female): ", ["male", "female"])
        # Append the sex to all the new subjects
        [new_subject.update({"sex": sex}) for new_subject in new_subjects]

        # Add date of birth
        while True:
            date_of_birth = input("Enter the date of birth (DD/MM/YYYY): ")
            try:
                datetime.strptime(date_of_birth, "%d/%m/%Y")
                [
                    new_subject.update({"date_of_birth": date_of_birth})
                    for new_subject in new_subjects
                ]
                break
            except ValueError:
                printme("Invalid date format. Please enter in DD/MM/YYYY format.")

        # Add cage number
        # first get the rows in which active is True
        active_subjects = subjects_data_csv[subjects_data_csv["active"] == True]
        active_existing_cage_numbers = active_subjects["cage_number"].unique().tolist()
        cage_number = self.get_input(
            "Enter the cage number or select from existing:",
            active_existing_cage_numbers + ["Enter new value"],
        )
        if cage_number == "Enter new value":
            cage_number = input("Enter new cage number: ")
        [
            new_subject.update({"cage_number": cage_number})
            for new_subject in new_subjects
        ]

        # Add species
        unique_species = genotypes_data_csv["species"].unique().tolist()
        species = self.get_input("Enter the species:", unique_species)
        [new_subject.update({"species": species}) for new_subject in new_subjects]

        # Add genotype
        genotypes_for_species = (
            genotypes_data_csv[genotypes_data_csv["species"] == species]["genotype"]
            .unique()
            .tolist()
        )
        genotype = self.get_input("Enter the genotype:", genotypes_for_species)
        [new_subject.update({"genotype": genotype}) for new_subject in new_subjects]

        # Default fields
        [
            new_subject.update({"current_license": "ZH_139"})
            for new_subject in new_subjects
        ]
        [new_subject.update({"current_subproject": ""}) for new_subject in new_subjects]
        [new_subject.update({"weight": ""}) for new_subject in new_subjects]
        [new_subject.update({"notes": ""}) for new_subject in new_subjects]
        [new_subject.update({"repository": ""}) for new_subject in new_subjects]
        [new_subject.update({"active": True}) for new_subject in new_subjects]
        [new_subject.update({"logged_out": False}) for new_subject in new_subjects]
        [new_subject.update({"logged_out_date": ""}) for new_subject in new_subjects]

        # Append new subjects to the CSV
        print(new_subjects)
        new_subjects_df = pd.DataFrame(new_subjects)
        subjects_data_csv = pd.concat(
            [subjects_data_csv, new_subjects_df], ignore_index=True
        )
        subjects_data_csv.to_csv(self.paths["subjects"], index=False)

        printme("New subjects added successfully.")

    def select_multiple_subjects(self):
        """
        Displays all subject IDs and allows the user to select multiple subjects.

        Returns:
            list: A list of selected subject IDs.
        """
        self.clear_input_buffer()
        subjects_data_dict = self.get_csv_data(self.paths["subjects"])
        subjects_data_dict = {
            k: v for k, v in subjects_data_dict.items() if v["active"] == True
        }
        subjects_options = [f"{key}" for key, _ in subjects_data_dict.items()]

        printme("Select the IDs of the subjects (separated by commas):")
        for i, option in enumerate(subjects_options):
            print(f"{i + 1}. {option}")

        while True:
            user_input = input(
                "Enter the numbers corresponding to the subjects, separated by commas: "
            )
            try:
                selected_indices = [int(x) - 1 for x in user_input.split(",")]
                selected_ids = [subjects_options[i] for i in selected_indices]
                print(selected_ids)
                return selected_ids
            except (ValueError, IndexError):
                printme(
                    "Invalid input, please enter valid numbers corresponding to the subjects."
                )

    @staticmethod
    def get_csv_data(file_path):
        """
        Find the specified CSV file, and extract
        its data into a nested dictionary where the first column's values
        are keys and the rest of the columns form the nested dictionary.

        Parameters:
            file_name (str): The name of the CSV file to read.

        Returns:
            dict: A nested dictionary where the first column's values are keys
                  and the rest of the columns form the nested dictionary.

        Raises:
            FileNotFoundError: If the specified CSV file does not exist in the
                               parent directory.
        """
        # Check if the file exists
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"The file {file_path} does not exist.")

        # Read the CSV file
        df = pd.read_csv(file_path)

        # Initialize the dictionary
        data_dict = {}

        # Populate the dictionary with the first column as keys and the rest as nested dictionaries
        first_column = df.columns[0]
        for _, row in df.iterrows():
            key = row[first_column]
            nested_dict = row.drop(first_column).to_dict()
            data_dict[key] = nested_dict

        return data_dict

    @staticmethod
    def get_input(prompt, options, start=1):
        """
        Gets input from the user, allowing selection from a list of options.

        Parameters:
            prompt (str): The prompt to display to the user.
            options (list): The list of options to choose from.

        Returns:
            str: The option selected by the user.
        """
        printme(prompt)
        for i, option in enumerate(options):
            print(f"{i + start}. {option}")

        while True:
            user_input = input(
                "Select an option by entering the corresponding number: "
            )
            if user_input.isdigit():
                index = int(user_input) - start
                if 0 <= index < len(options):
                    return options[index]
            printme(
                "Invalid input, please enter a number corresponding to the options above."
            )

    @staticmethod
    def append_timestamp(data):
        """
        Appends the current timestamp to the given data.

        Parameters:
            data (list): The data to which the timestamp will be appended.

        Returns:
            list: The data with the current timestamp appended at the beginning.
        """
        current_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return [current_timestamp] + data

    @staticmethod
    def clear_input_buffer():
        """
        Clears the input buffer to prevent unexpected behavior.
        """
        if platform.system() == "Windows":
            import msvcrt

            while msvcrt.kbhit():
                msvcrt.getch()
        else:
            import sys, termios

            termios.tcflush(sys.stdin, termios.TCIOFLUSH)


def show_alert(subject_id, message):
    """
    Display an alert window with a warning message.

    Parameters:
    subject_id (str): The ID of the subject.
    message (str): The warning message to display.
    """
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    messagebox.showwarning(f"Warning for Subject {subject_id}", message)
    root.destroy()
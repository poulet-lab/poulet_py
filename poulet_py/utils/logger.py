import csv
from datetime import datetime
import os
import pandas as pd
import os
import ast
import platform

def printme(message):
    print(f"\n{message}\n")

class SessionLogger:
    """
    A class to log experimental sessions by collecting various data points from the user
    and writing them to a logbook.
    """

    def __init__(self, path):
        self.file_names = [
            'logbook.csv', 'subjects.csv', 'licenses.csv',
            'methods.csv', 'experimenters.csv', 'experimental_designs.csv',
            'genotypes.csv'
        ]
        self.path = path
        self.paths = {
            os.path.splitext(file_name)[0]: os.path.join(path, file_name) for file_name in self.file_names
        }

        self.subject_id = None
        self.license = None
        self.subproject = None
        self.method = None
        self.method_version = None
        self.duration_s = None
        self.condition = None
        self.experimenter = None
        self.notes = None

        self.clear_input_buffer()


    def get_subject_id(self):
        """
        Prompts user to enter the subject ID and retrieves the corresponding subject number.
        """
        subjects_data_dict = self.get_csv_data(self.paths['subjects'])
        #remove the subjects in which active is False
        subjects_data_dict = {k: v for k, v in subjects_data_dict.items() if v['active'] == True}
        subjects_options = [f"{key}" for key, _ in subjects_data_dict.items()]
        subject_id = self.get_input("Enter the ID of the subject", subjects_options, start=0)
        self.subject_id = subject_id.split()[0]

        printme(f"Subject ID: {self.subject_id}")


    def get_license_data(self):
        """
        Retrieves the license data for the subject.
        """

        self.license = self.get_current_license()
        if self.license in ['ZH_139', 'X9016_21', 'G0167_23']:
            license_data = self.get_csv_data(self.paths['licenses'])
            self.license = self.get_input("Enter the license", list(license_data.keys()))
            # update the current license in the subjects.csv file
            subjects_data_csv = pd.read_csv(self.paths['subjects'])
            subjects_data_csv.loc[subjects_data_csv['subject_id'] == self.subject_id, 'current_license'] = self.license
            subjects_data_csv.to_csv(self.paths['subjects'], index=False)

        printme(f"License: {self.license}")


    def get_subproject_data(self):
        """
        Retrieves the subproject data for the subject.
        """

        self.subproject = self.get_current_subproject()
        if self.subproject is None:
            license_data = self.get_csv_data(self.paths['licenses'])
            subprojects = eval(license_data[self.license]['subprojects'])
            self.subproject = self.get_input("Enter the subproject", subprojects)
            # update the current subproject in the subjects.csv file
            subjects_data_csv = pd.read_csv(self.paths['subjects'])
            subjects_data_csv.loc[subjects_data_csv['subject_id'] == self.subject_id, 'current_subproject'] = self.subproject
            subjects_data_csv.to_csv(self.paths['subjects'], index=False)

        printme(f"Subproject: {self.subproject}")
    

    def get_method_data(self):
        """
        Prompts user to select method and returns it.
        """
        if self.method is None:
            method_data = self.get_csv_data(self.paths['methods'])
            self.method = self.get_input("Enter the method", list(method_data.keys()))

        printme(f"Method: {self.method}")


    def get_method_version_data(self):
        """
        Prompts user to select method version and returns it.
        """
        if self.method_version is None:
            method_data = self.get_csv_data(self.paths['methods'])
            method_versions = eval(method_data[self.method]['versions'])
            self.method_version = self.get_input("Enter the version", method_versions)
        
        printme(f"Method version: {self.method_version}")


    def get_experimenter_data(self):
        """
        Prompts user to select experimenter and returns it.
        """
        if self.experimenter is None:
            experimenter_data = self.get_csv_data(self.paths['experimenters'])
            self.experimenter = self.get_input("Enter the experimenter", list(experimenter_data.keys()))

        printme(f"Experimenter: {self.experimenter}")


    def get_condition_data(self):
        """
        Retrieves the condition data for the subject.
        """
        self.condition = self.get_mouse_condition()
        if self.condition is None:
            conditions_data_csv = pd.read_csv(self.paths['experimental_designs'])
            #get the rows with license_numner is self.license
            license_rows = conditions_data_csv[conditions_data_csv['license_number'] == self.license]
            #get the rows with subproject is self.subproject
            subproject_rows = license_rows[license_rows['subproject'] == self.subproject]
            #get the values in the column condition
            condition_data = subproject_rows['condition'].unique().tolist()

            print(condition_data)
            self.condition = self.get_input("Enter the condition", list(condition_data))

        printme(f"Condition: {self.condition}")


    def get_duration_data(self):
        """
        Prompts user to enter duration of the experiment and returns it.
        """
        if self.duration_s is None:
            duration = input("Enter the duration of the experiment (in seconds): ")
            self.duration_s = int(duration) if duration.isdigit() else None

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

        self.get_subject_id()

        self.get_method_data()

        if self.method == 'weighing':
            self.log_weight()
            return

        self.get_license_data()
        
        self.get_subproject_data()

        self.get_method_version_data()

        self.get_experimenter_data()

        self.get_condition_data()

        self.get_duration_data()

        self.get_notes_data()


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


    def log_weight(self):
        """
        Logs the weight of the subject by adding an entry to the weight logbook.
        """
        self.clear_input_buffer()
        subjects_data_csv = pd.read_csv(self.paths['subjects'])
        # Subject ID
        if self.subject_id is None:
            subjects_data_dict = self.get_csv_data(self.paths['subjects'])            
            subjects_options = [f"{key}" for key, _ in subjects_data_dict.items()]
            subject_id = self.get_input("Enter the ID of the subject", subjects_options, start=0)
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
        date = datetime.now().strftime('%d/%m/%Y')

        # Add the weight entry to the CSV
        # check whether there's a row for the subject_id in the CSV
        if subjects_data_csv['subject_id'].isin([self.subject_id]).any():
            # Retrieve the current cell value
            current_value = subjects_data_csv.loc[subjects_data_csv['subject_id'] == self.subject_id, 'weight'].iloc[0]
            
            # Check if the cell is not empty and contains a dictionary
            if pd.notna(current_value):
                # Convert the string back to a dictionary
                current_dict = ast.literal_eval(current_value)
            else:
                # If the cell is empty, initialize a new dictionary
                current_dict = {}
            
            # Add the new date and weight
            current_dict[date] = self.weight
            
            # Convert the dictionary back to a string and update the DataFrame
            subjects_data_csv.loc[subjects_data_csv['subject_id'] == self.subject_id, 'weight'] = str(current_dict)

            # write the updated data to the CSV
            subjects_data_csv.to_csv(self.paths['subjects'], index=False)

            self.license = self.get_current_license()
            self.subproject = self.get_current_subproject()
            self.method = 'weighing'
            self.method_version = '101'
            self.duration_s = 60
            self.condition = self.get_mouse_condition()
            self.experimenter = 'IER'
            self.notes = f"Weight of {str(self.weight)} grams"

            print(f"Weight of {self.weight} grams logged for subject {self.subject_id} on {date}.")
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
        df = pd.read_csv(self.paths['subjects'])
        # Search for the subject_id and get the corresponding license
        subject_row = df[df['subject_id'] == self.subject_id]
        
        if not subject_row.empty:
            return subject_row.iloc[0]['current_license']
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
        df = pd.read_csv(self.paths['subjects'])
        # Search for the subject_id and get the corresponding subproject
        subject_row = df[df['subject_id'] == self.subject_id]

        if not pd.isnull(subject_row.iloc[0]['current_subproject']):
            return subject_row.iloc[0]['current_subproject']
        else:
            return None
  

    def get_mouse_condition(self):
        """
        Given a subject_id, output the corresponding condition from the CSV file.

        Returns:
            str: The condition corresponding to the provided subject_id.
            None: If the subject_id is not found.
        """

        if self.license != 'ZH_139' or self.license != 'X9016_21' or self.license != 'G0167_23':

            # Read the CSV file into a DataFrame
            df = pd.read_csv(self.paths['experimental_designs'])

            # Get the rows in which the license_number is self.license
            license_rows = df[df['license_number'] == self.license]

            # Get the rows in which the subproject is self.subproject
            subproject_rows = license_rows[license_rows['subproject'] == self.subproject]
            print(subproject_rows)
            # Convert the 'subjects' column from string to list
            subproject_rows['subjects'] = subproject_rows['subjects'].apply(ast.literal_eval)

            subject_row = subproject_rows[subproject_rows['subjects'].apply(lambda x: self.subject_id in x)]

            print(self.subject_id)

            if not subject_row.empty:
                return subject_row.iloc[0]['condition']
            else:
                return None


    def log_session(self):
        """
        Logs the session by adding an entry to the logbook.
        """
        
        if self.subject_id is None:
            raise ValueError("subject_id must be provided.")
    
        os.makedirs(os.path.dirname(self.paths['logbook']), exist_ok=True)

        log_entry_data = [
            str(self.subject_id), str(self.license), 
            str(self.subproject), str(self.method), str(self.method_version), str(self.duration_s),
            str(self.condition), str(self.experimenter), str(self.notes)
        ]
        log_entry_with_timestamp = self.append_timestamp(log_entry_data)

        try:
            file_exists = os.path.isfile(self.paths['logbook'])
            with open(self.paths['logbook'], mode='a', newline='') as file:
                writer = csv.writer(file)
                if not file_exists:
                    writer.writerow(['timestamp', 'subject_id', 'license', 'subproject', 'method', 'method_version', 'duration_s', 'condition', 'experimenter', 'notes'])
                writer.writerow(log_entry_with_timestamp)
            print(f"Log entry added: {log_entry_with_timestamp}")
        except Exception as e:
            print(f"Error adding log entry: {e}")


    def add_subjects(self):
        """
        Adds new subjects to the subjects.csv file.
        """
        self.clear_input_buffer()
        subjects_data_csv = pd.read_csv(self.paths['subjects'])
        genotypes_data_csv = pd.read_csv(self.paths['genotypes'])

        new_subjects = []

        while True:

            # Add subject ID
            while True:
                subject = {}
                subject_id = input("Enter the subject ID: ")
                confirmation = input(f"Confirm subject ID '{subject_id}' (y/n): ").lower()
                if confirmation == 'y':
                    #append the subject_id to the subject dictionary
                    subject['subject_id'] = subject_id
                    new_subjects.append(subject)
                    break
                elif confirmation == 'n':
                    continue
                else:
                    printme("Invalid input. Please enter 'y' for yes or 'n' for no.")

            # Confirm adding more subjects
            add_more_ids = input("Do you want to add more IDs for the same cage? (y/n): ").lower()
            if add_more_ids == 'n':
                break


        # Add sex
        sex = self.get_input("Enter the sex (male/female): ", ["male", "female"])
        # Append the sex to all the new subjects
        [new_subject.update({'sex': sex}) for new_subject in new_subjects]

        # Add date of birth
        while True:
            date_of_birth = input("Enter the date of birth (DD/MM/YYYY): ")
            try:
                datetime.strptime(date_of_birth, '%d/%m/%Y')
                [new_subject.update({'date_of_birth': date_of_birth}) for new_subject in new_subjects]
                break
            except ValueError:
                printme("Invalid date format. Please enter in DD/MM/YYYY format.")

        # Add cage number
        existing_cage_numbers = subjects_data_csv['cage_number'].unique().tolist()
        cage_number = self.get_input("Enter the cage number or select from existing:", existing_cage_numbers + ["Enter new value"])
        if cage_number == "Enter new value":
            cage_number = input("Enter new cage number: ")
        [new_subject.update({'cage_number': cage_number}) for new_subject in new_subjects]

        # Add species
        unique_species = genotypes_data_csv['species'].unique().tolist()
        species = self.get_input("Enter the species:", unique_species)
        [new_subject.update({'species': species}) for new_subject in new_subjects]

        # Add genotype
        genotypes_for_species = genotypes_data_csv[genotypes_data_csv['species'] == species]['genotype'].unique().tolist()
        genotype = self.get_input("Enter the genotype:", genotypes_for_species)
        [new_subject.update({'genotype': genotype}) for new_subject in new_subjects]

        # Default fields
        [new_subject.update({'current_license': 'ZH_139'}) for new_subject in new_subjects]
        [new_subject.update({'current_subproject': ''}) for new_subject in new_subjects]
        [new_subject.update({'weight': ''}) for new_subject in new_subjects]
        [new_subject.update({'notes': ''}) for new_subject in new_subjects]
        [new_subject.update({'repository': ''}) for new_subject in new_subjects]
        
        # Append new subjects to the CSV
        print(new_subjects)
        new_subjects_df = pd.DataFrame(new_subjects)
        subjects_data_csv = pd.concat([subjects_data_csv, new_subjects_df], ignore_index=True)
        subjects_data_csv.to_csv(self.paths['subjects'], index=False)

        printme("New subjects added successfully.")


    def select_multiple_subjects(self):
        """
        Displays all subject IDs and allows the user to select multiple subjects.

        Returns:
            list: A list of selected subject IDs.
        """
        self.clear_input_buffer()
        subjects_data_dict = self.get_csv_data(self.paths['subjects'])
        subjects_data_dict = {
            k: v for k, v in subjects_data_dict.items() if v["active"] == True
        }
        subjects_options = [f"{key}" for key, _ in subjects_data_dict.items()]

        printme("Select the IDs of the subjects (separated by commas):")
        for i, option in enumerate(subjects_options):
            print(f"{i + 1}. {option}")

        while True:
            user_input = input("Enter the numbers corresponding to the subjects, separated by commas: ")
            try:
                selected_indices = [int(x) - 1 for x in user_input.split(',')]
                selected_ids = [subjects_options[i] for i in selected_indices]
                print(selected_ids)
                return selected_ids
            except (ValueError, IndexError):
                printme("Invalid input, please enter valid numbers corresponding to the subjects.")


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
    def get_input(prompt, options, start = 1):
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
            user_input = input("Select an option by entering the corresponding number: ")
            if user_input.isdigit():
                index = int(user_input) - start
                if 0 <= index < len(options):
                    return options[index]
            printme("Invalid input, please enter a number corresponding to the options above.")


    @staticmethod
    def append_timestamp(data):
        """
        Appends the current timestamp to the given data.

        Parameters:
            data (list): The data to which the timestamp will be appended.

        Returns:
            list: The data with the current timestamp appended at the beginning.
        """
        current_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        return [current_timestamp] + data
    
    @staticmethod
    def clear_input_buffer():
        """
        Clears the input buffer to prevent unexpected behavior.
        """
        if platform.system() == 'Windows':
            import msvcrt
            while msvcrt.kbhit():
                msvcrt.getch()
        else:
            import sys, termios
            termios.tcflush(sys.stdin, termios.TCIOFLUSH)
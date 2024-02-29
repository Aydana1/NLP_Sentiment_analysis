import streamlit as st
import PyPDF2

import tabula
from tabulate import tabulate
import pandas as pd

from pdf2image import convert_from_path
import io
import re
from PIL import Image
import time

import PDF_doc

st.set_page_config(layout="wide")


def read_pdf_tabula(pdf_path):
    # Use tabula to extract tables from the PDF
    tables = tabula.read_pdf(pdf_path, pages='1', multiple_tables=False) #area=[100.51, 21.51, 564.62, 595.92]
    df_tables = pd.DataFrame((tables[0]))

    return df_tables

city_country_to_continent = {
    "Paris": "Europe",
    "London": "Europe",
    "England": "Europe",
    "New York City": "North America",
    "Zimbabwe": "South Africa",
    "Seattle": "North America",
    "Stockholm": "Europe",
    "Turkey": "Middle East"
}

def get_continent_manual(city_or_country):
    return city_country_to_continent[city_or_country]

def pdf_to_images(pdf_path):
    images = convert_from_path(pdf_path, 300)  # 200 DPI, adjust as needed
    return images

def main():
    # Set the title of the app
    st.title("PDF Analyzer")

    # Create a file uploader widget
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    # Check if a file has been uploaded
    if uploaded_file is not None:
        # Read the contents of the PDF file
        file_name = uploaded_file.name
        df = read_pdf_tabula(file_name)
        str_indx = [0, 5, 11, 15, 19, 25, 29, 32, 34]
      
        #Explain the goal
        st.write("PDF has a table with information of individuals who have responded to a survey in different languages. \
                     The individuals have answered the following: “Please introduce yourself, specify where you live, and tell us what is your opinion on the city of Paris?”")

        pdf_images = pdf_to_images(file_name)
        col1, col2 = st.columns(2)
        # Display each page as an image
        for i, image in enumerate(pdf_images):
            new_image = image.resize((800, 1000))
            with col1:
                  # Display the content of the PDF
                # st.subheader("PDF Content:")
                st.image(new_image, caption=f"Page {i + 1}") 

        # Process the PDF
        pdf_doc1 = PDF_doc.PDF_doc()
        pdf_path = file_name
        str_arr, date_arr, time_arr, gender_arr = pdf_doc1.df_to_list(str_indx, df)
        i = 0
        answers_arr = []
        names_arr = []
        cities_arr = []
        ages_arr = []
        sentiment_arr = []
        rudeness_arr = []
        continents_arr = []
        
        for input in str_arr:
            i += 1
            translation, person_names, person_locations, ages, blob_score, vader_score, rudeness_score = pdf_doc1.process_pdf(input, i)
            #remove Paris city names once from list of strings (for people living in Paris)
            for city in person_locations:
                if city == 'Paris':
                    person_locations.remove(city)

            for loc in person_locations:
                continents_arr.append(get_continent_manual(loc))
                # print(get_continent_manual(loc))
                break
            
            person_names = ''.join(person_names)
            person_locations = ', '.join(person_locations)
        
            ages = ''.join(ages)
            vader_score = "{:.3f}".format(vader_score)
            rudeness_score =  "{:.3f}".format(rudeness_score["toxicity"])

            answers_arr.append(translation)
            names_arr.append(person_names)
            cities_arr.append(person_locations)
            ages_arr.append(ages)
            sentiment_arr.append(vader_score)
            rudeness_arr.append(rudeness_score)

            with col2:  
                with st.spinner('In progress...'):
                    time.sleep(5) 
        
        print(continents_arr)

        output_data = {
            'Date': date_arr,
            'Time': time_arr,
            'Gender': gender_arr,
            'Name': names_arr,
            'Loc': continents_arr,
            'Age': ages_arr,
            'Opinion Score': sentiment_arr,
            'Rudeness Score': rudeness_arr,
            'Answer': answers_arr  
        }

        output_df = pd.DataFrame(output_data)
        output_df.set_index('Date', inplace=True)
        output_df.sort_values(by='Opinion Score', ascending=False, inplace=True)
        with col2:
            st.subheader("Sentiment Analysis Result:")
            st.table(output_df)


if __name__ == "__main__":
    main()

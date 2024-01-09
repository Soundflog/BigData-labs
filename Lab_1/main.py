import csv
import re

from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd


def TfidVectorized_csv(df):
    v = TfidfVectorizer(min_df=5)
    x = v.fit_transform(df)
    print(v.get_feature_names_out())
    print(x.shape)
    array = x.toarray()
    print(array[array != 0])

    # dense = x.to_dense()
    # dense_list = dense.to_list()
    #
    # all_keywords = []
    #
    # for description in dense_list:
    #     x = 0
    #     keywords = []
    #     for word in description:
    #         if word > 0:
    #             keywords.append(v.get_feature_names_out()[x])
    #         x = x + 1
    #     all_keywords.append(keywords)
    #
    # print(all_keywords[0])


def Transforming(data_train, data_test):
    enc = DictVectorizer

    X_train_categ = enc.fit_transform(
        data_train[['LocationNormalized', 'ContractTime']]
        .to_dict('records'), 2
    )
    print(X_train_categ.toarray())

    # X_test_categ = enc.transform(
    #     data_test[['LocationNormalized', 'ContractTime']]
    #     .to_dict('records')
    # )


def Post_Update_Data(csv_file, param):
    # 1 variant

    # train_csv_file = open(csv_file)
    # train_reader = csv.reader(train_csv_file)
    # new_list_train = []
    # for line in train_reader:
    #     for i in line:
    #         new_list_train.append(re.findall('[^a-zA-Z0-9]', i.lower()))
    # train_df = pd.DataFrame(new_list_train)
    # print(train_df)
    #
    # test_csv_file = open(csv_file_2)
    # test_reader = csv.reader(test_csv_file)
    # new_list_test = []
    # for line in test_reader:
    #     for i in line:
    #         new_list_test.append(re.findall('[^a-zA-Z0-9]',  i.lower()))
    # test_df = pd.DataFrame(new_list_test)
    # print(test_df)

    # 2 variant

    # train_csv = pd.read_csv(csv_file, quotechar='"')
    # test_csv = pd.read_csv(csv_file_2)
    # train_csv.apply(lambda x: re.sub('[^a-zA-Z0-9]', ' ', x.lower()))
    # test_csv.apply(lambda x: re.sub('[^a-zA-Z0-9]', ' ', x.lower()))
    # re.sub('[^a-zA-Z0-9]', ' ', train_csv.lower())
    # re.sub('[^a-zA-Z0-9]', ' ', test_csv.lower())

    # 3 variant
    data = []
    with open(csv_file, 'r', encoding='utf-8', newline='') as file:
        reader = csv.DictReader(file)
        for row in reader:
            # print(row["FullDescription"])
            row[param] = re.sub('[^a-zA-Z0-9]', ' ', row[param].lower())
            data.append(row[param])

    # df = pd.DataFrame(data)
    return data


if __name__ == '__main__':
    csv_name_file = "salary-train.csv"
    csv_name_file_2 = "salary-test-mini.csv"
    # df_train_csv = pd.read_csv(csv_name_file)
    # for row in df_train_csv[['FullDescription']]:
    #     row = re.sub('[^a-zA-Z0-9]', ' ', row.lower())
    # print(df_train_csv.head())

    # des_train_csv_clean = Post_Update_Data(csv_name_file, "FullDescription")
    # TfidVectorized_csv(des_train_csv_clean)

    # print(train_csv_post_update_data)

    # Transforming(, salary_test_csv)

    # string_test = "International Sales Manager London ****k  ****k  Uncapped Commission Digital Marketing/Performance Marketing A highly ambitious Performance Marketing company and one of the leaders in the European market are seeking excellent Business Development/Sales professionals for their International team based in London. With approximately **** clients across a number of verticals including travel tourism, telecoms, financial services and retail their EU reach incorporates offices in 17 countries, with this soon to be expanded to **** There are further plans to set up in Asia and the emerging Middle Eastern markets. Clients include some of the largest companies in telecoms, finance, travel and computer software. The International Sales Manager will ultimately be responsible for growing the global client base by pitching and winning new business with large, bluechip, international organisations. This will involve worldwide travel to client offices mainly in Europe but also across Asia and parts of the Middle East. Responsibilities:  Generating sales leads and making initial sales calls to potential clients  Preparing pitches/sales presentations using PowerPoint or other relevant tools  Attending sales meetings/pitches alongside the group CEO and/or COO  Drawing up business proposals and contracts  International travel to client offices and/or other offices  Forecasting sales plans and revenue targets/figures Personal specification/skills required:  Personable candidate who demonstrates outstanding enthusiasm, selfbelief and relationshipbuilding skills  Previous sales experience in the Digital Marketing sector (preferably in performance/affiliate marketing)  Comfortable and confident in both phone based and consultative facetoface sales situations  Competence to create and deliver structured presentations and proposals  Be tenacious and driven to succeed at all times  in a highly competitive and challenging environment  Ability to generate own sales opportunities and think laterally to succeed  Be a highly results focused and target driven individual who is able to build a continually evolving pipeline and provide accurate forecasts  Educated ideally to degree level standard  Excellent fluent spoken and written English  The ability to speak an additional language, preferably French, would be a huge bonus Essentially we are looking for an extremely focused and highly career driven individual with the ability to sell marketing leading technology at the highest level."
    # string_test = re.sub('[^a-zA-Z0-9]', ' ', string_test.lower())
    # print(string_test)

import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, RANSACRegressor, BayesianRidge, PoissonRegressor
from sklearn.svm import SVR  
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor 
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsRegressor
from statsmodels.tsa.arima_model import ARIMA
import statsmodels.api as sm
import pandas.plotting as pd_plotting
from sktime.forecasting.exp_smoothing import ExponentialSmoothing
from prophet import Prophet
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import RANSACRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, explained_variance_score, precision_score, recall_score, roc_auc_score, accuracy_score, f1_score, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis
from sklearn.linear_model import Perceptron, PassiveAggressiveClassifier, SGDClassifier
from sklearn.neighbors import RadiusNeighborsClassifier, NearestCentroid
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.svm import NuSVC
from sklearn.semi_supervised import LabelPropagation, LabelSpreading
from sklearn.ensemble import ExtraTreesClassifier
import pickle
import stemgraphic
import joypy
import requests
import scipy.stats as stats
import plotly.io as pio
from streamlit_lottie import st_lottie

st.set_page_config(
    page_title="Data Voyager",
    page_icon="🤖",
    layout="wide"
)
st.set_option('deprecation.showPyplotGlobalUse', False)

def main():
    st.title("Data Voyager 🤖")
    st.write("Welcome to the Streamlit EDA App! This app allows you to perform Exploratory Data Analysis (EDA), visualize data, and build machine learning models.")

    data = None
    # Upload dataset or select sample data
    data_source = st.sidebar.radio("Select Data Source", ["Upload Dataset", "Use Sample Data"])
    if data_source == "Upload Dataset":
        uploaded_file = st.sidebar.file_uploader("Upload your dataset (CSV)", type=["csv"])
        if uploaded_file:
            data = pd.read_csv(uploaded_file, low_memory=False)
            st.write("### Sneak peak of the data")
            st.write(data.head(10))
    else:
        sample_dataset_name = st.sidebar.selectbox("Select a sample dataset", ["tips", "titanic", "penguins", "iris", "diamonds", "planets", "fmri", "attention", "car_crashes", "flights"])
        st.write("### Sneak peak of the data")
        data = sns.load_dataset(sample_dataset_name)
        st.write(data.head())

    if data is not None:
        # EDA Section
        if st.sidebar.checkbox("Do you want to perform basic EDA analysis?"):
            st.write("### Dataset Shape:")
            st.write(data.shape)
            st.write("### Dataset Columns:")
            st.write(data.columns)
            st.write("### Dataset Data Types:")
            st.write(data.dtypes)
            st.write("### Dataset Missing Values:")
            st.write(data.isnull().sum())
            st.write("### Dataset Unique Values:")
            st.write(data.nunique())
            # st.write("### Dataset Skewness:")
            # st.write(data.skew())
            # st.write("### Dataset Kurtosis:")
            # st.write(data.kurtosis())
            # st.write("### Dataset Standard Deviation:")
            # st.write(data.std())
            # st.write("### Dataset Variance:")
            # st.write(data.var())
            # st.write("### Dataset Quantiles:")
            # st.write(data.quantile([0.25, 0.5, 0.75]))
            st.write("### Summary Statistics:")
            st.write(data.describe())
            if len(data.columns) < 5:
                st.write("### Pairplot:")
                fig = sns.pairplot(data)
                st.pyplot(fig)
            else:
                st.write("### Correlation Matrix:")
                numeric_data = data.select_dtypes(include=[np.number])
                corr_matrix = numeric_data.corr()
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(corr_matrix, annot=True, ax=ax, cmap="coolwarm")
                st.pyplot(fig)


        # Missing Value Imputation
        if st.sidebar.checkbox("Do you want to impute missing values?"):
            st.sidebar.subheader("Imputation Options")
            missing_value_techs = ["Mean", "Median", "Mode", "Zero", "KNN", "Regression Imputation", "Random Forest Imputation"]
            missing_value_choice = st.sidebar.selectbox("Choose an imputation method", missing_value_techs)

            numeric_data = data.select_dtypes(include=[np.number])
            if missing_value_choice == "Mean":
                imputer = SimpleImputer(strategy='mean')
            elif missing_value_choice == "Median":
                imputer = SimpleImputer(strategy='median')
            elif missing_value_choice == "Mode":
                imputer = SimpleImputer(strategy='most_frequent')
            elif missing_value_choice == "Zero":
                imputer = SimpleImputer(strategy='constant', fill_value=0)
            elif missing_value_choice == "KNN":
                imputer = KNNImputer(n_neighbors=5)
            elif missing_value_choice == "Regression Imputation":
                imputer = IterativeImputer(estimator=LinearRegression(), max_iter=10, random_state=0)
            elif missing_value_choice == "Random Forest Imputation":
                imputer = IterativeImputer(estimator=RandomForestRegressor(), max_iter=10, random_state=0)
            
            data[numeric_data.columns] = imputer.fit_transform(data[numeric_data.columns])
            st.write(f"Missing values imputed using {missing_value_choice} method.")
            st.write("### Dataset after Imputation:")
            st.write(data.head())

        # Encoding Section
        st.sidebar.subheader("Data Encoding")
        encoding_method = st.sidebar.selectbox("Select an encoding method", ["None", "One-Hot Encoding", "Label Encoding", "Binary Encoding", "Hash Encoding", "BaseN Encoding", "Backward Difference Encoding", "Helmert Encoding", "Sum Encoding", "Polynomial Encoding", "Leave One Out Encoding", "CatBoost Encoding", "James-Stein Encoding", "M-estimator Encoding", "Target Encoding", "Weight of Evidence Encoding", "GLMM Encoding", "WOE Encoding", "Rare Encoding", "Count Encoding", "Probability Ratio Encoding", "Feature Embedding"])

        if encoding_method == "One-Hot Encoding":
            data = pd.get_dummies(data)
            st.write("Data after One-Hot Encoding:")
            st.write(data.head())
        elif encoding_method == "Label Encoding":
            label_encoder = LabelEncoder()
            for col in data.columns:
                if data[col].dtype == "object":
                    data[col] = label_encoder.fit_transform(data[col])
            st.write("Data after Label Encoding:")
            st.write(data.head())
        elif encoding_method == "Binary Encoding":
            from category_encoders import BinaryEncoder
            encoder = BinaryEncoder()
            data = encoder.fit_transform(data)
            st.write("Data after Binary Encoding:")
            st.write(data.head())
        elif encoding_method == "Hash Encoding":
            from category_encoders import HashingEncoder
            encoder = HashingEncoder()
            data = encoder.fit_transform(data)
            st.write("Data after Hash Encoding:")
            st.write(data.head())
        elif encoding_method == "BaseN Encoding":
            from category_encoders import BaseNEncoder
            encoder = BaseNEncoder()
            data = encoder.fit_transform(data)
            st.write("Data after BaseN Encoding:")
            st.write(data.head())


        # Scaling Section
        st.sidebar.subheader("Data Scaling")
        scaling_method = st.sidebar.selectbox("Select a scaling method", ["None", "Standard Scaling", "Min-Max Scaling", "Max Abs Scaling", "Robust Scaling", "Quantile Transformation (Uniform)", "Quantile Transformation (Gaussian)", "Power Transformation (Yeo-Johnson)", "Power Transformation (Box-Cox)"])

        if scaling_method == "Standard Scaling":
            scaler = StandardScaler()
            data[X_columns] = scaler.fit_transform(data[X_columns])
            st.write("Data after Standard Scaling:")
            st.write(data.head())
        elif scaling_method == "Min-Max Scaling":
            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler()
            data[X_columns] = scaler.fit_transform(data[X_columns])
            st.write("Data after Min-Max Scaling:")
            st.write(data.head())
        elif scaling_method == "Max Abs Scaling":
            from sklearn.preprocessing import MaxAbsScaler
            scaler = MaxAbsScaler()
            data[X_columns] = scaler.fit_transform(data[X_columns])
            st.write("Data after Max Abs Scaling:")
            st.write(data.head())
        elif scaling_method == "Robust Scaling":
            from sklearn.preprocessing import RobustScaler
            scaler = RobustScaler()
            data[X_columns] = scaler.fit_transform(data[X_columns])
            st.write("Data after Robust Scaling:")
            st.write(data.head())
        elif scaling_method == "Quantile Transformation (Uniform)":
            from sklearn.preprocessing import QuantileTransformer
            scaler = QuantileTransformer(output_distribution='uniform')
            data[X_columns] = scaler.fit_transform(data[X_columns])
            st.write("Data after Quantile Transformation (Uniform):")
            st.write(data.head())
        elif scaling_method == "Quantile Transformation (Gaussian)":
            from sklearn.preprocessing import QuantileTransformer
            scaler = QuantileTransformer(output_distribution='normal')
            data[X_columns] = scaler.fit_transform(data[X_columns])
            st.write("Data after Quantile Transformation (Gaussian):")
            st.write(data.head())
        elif scaling_method == "Power Transformation (Yeo-Johnson)":
            from sklearn.preprocessing import PowerTransformer
            scaler = PowerTransformer(method='yeo-johnson')
            data[X_columns] = scaler.fit_transform(data[X_columns])
            st.write("Data after Power Transformation (Yeo-Johnson):")
            st.write(data.head())
        elif scaling_method == "Power Transformation (Box-Cox)":
            from sklearn.preprocessing import PowerTransformer
            scaler = PowerTransformer(method='box-cox')
            data[X_columns] = scaler.fit_transform(data[X_columns])
            st.write("Data after Power Transformation (Box-Cox):")
            st.write(data.head())

        # Feature Selection Section
        st.sidebar.subheader("Feature Selection")
        feature_selection_method = st.sidebar.selectbox("Select a feature selection method", ["None", "SelectKBest", "Recursive Feature Elimination (RFE)", "Recursive Feature Elimination with Cross-Validation (RFECV)", "SelectFromModel"])

        if feature_selection_method == "SelectKBest":
            from sklearn.feature_selection import SelectKBest, f_regression
            k = st.sidebar.slider("Select the number of features", 1, len(X_columns), 3)
            selector = SelectKBest(f_regression, k=k)
            selector.fit(X_train, y_train)
            selected_features = X_train.columns[selector.get_support()]
            st.write("Selected Features:")
            st.write(selected_features)
            X_train = selector.transform(X_train)
            X_test = selector.transform(X_test)
        elif feature_selection_method == "Recursive Feature Elimination (RFE)":
            from sklearn.feature_selection import RFE
            estimator = LinearRegression()
            selector = RFE(estimator, n_features_to_select=3, step=1)
            selector.fit(X_train, y_train)
            selected_features = X_train.columns[selector.get_support()]
            st.write("Selected Features:")
            st.write(selected_features)
            X_train = selector.transform(X_train)
            X_test = selector.transform(X_test)
        elif feature_selection_method == "Recursive Feature Elimination with Cross-Validation (RFECV)":
            from sklearn.feature_selection import RFECV
            estimator = LinearRegression()
            selector = RFECV(estimator, step=1, cv=5)
            selector.fit(X_train, y_train)
            selected_features = X_train.columns[selector.get_support()]
            st.write("Selected Features:")
            st.write(selected_features)
            X_train = selector.transform(X_train)
            X_test = selector.transform(X_test)
        elif feature_selection_method == "SelectFromModel":
            from sklearn.feature_selection import SelectFromModel
            estimator = LinearRegression()
            selector = SelectFromModel(estimator)
            selector.fit(X_train, y_train)
            selected_features = X_train.columns[selector.get_support()]
            st.write("Selected Features:")
            st.write(selected_features)
            X_train = selector.transform(X_train)
            X_test = selector.transform(X_test)


        # Plotting Section
        st.sidebar.subheader("Plotting Options")

        # Allow users to choose plots by different categories
        plot_categories = st.sidebar.selectbox("Select Plot Category", ["Univariate", "Bivariate", "Multivariate"])

        if "Univariate" in plot_categories:
            st.sidebar.subheader("Univariate Plots")
            columns_to_plot = st.sidebar.multiselect("Select only 1 column for Univariate Plots", data.columns)
            
            if len(columns_to_plot) == 1:
                column = columns_to_plot[0]
                
                if data[column].dtype in [np.float64, np.int64]:
                    # Allow users to choose which plot to display
                    plot_options = ["Histogram", "Density Plot", "Box Plot", "Violin Plot", "Dot Plot", "Stem & Leaf Plot", "Strip Plot", "Ridgeline Plot", "Joy Plot", "Cumulative Distribution Plot", "Q-Q Plot", "Probability Plot"]
                    plot_choice = st.sidebar.selectbox("Select a plot to display", plot_options)
                    
                    # Plot the selected plot
                    if plot_choice == "Histogram":
                        fig_hist = px.histogram(data, x=column, title=f"Histogram: {column}")
                        st.plotly_chart(fig_hist)
                    elif plot_choice == "Density Plot":
                        fig_density = px.density_contour(data, x=column, title=f"Density Plot: {column}")
                        st.plotly_chart(fig_density)
                    elif plot_choice == "Box Plot":
                        fig_box = px.box(data, y=column, title=f"Box Plot: {column}")
                        st.plotly_chart(fig_box)
                    elif plot_choice == "Violin Plot":
                        fig_violin = px.violin(data, y=column, title=f"Violin Plot: {column}")
                        st.plotly_chart(fig_violin)
                    elif plot_choice == "Dot Plot":
                        plt.figure()
                        plt.plot(data[column], 'ro', alpha=0.5)
                        plt.title(f"Dot Plot: {column}")
                        st.pyplot(plt)
                    elif plot_choice == "Stem & Leaf Plot":
                        stem_data = data[column].dropna().tolist()
                        sg = stemgraphic.stem_graphic(stem_data)
                        st.write(f"Stem & Leaf Plot: {column}")
                        st.pyplot(sg)
                    elif plot_choice == "Strip Plot":
                        fig_strip = px.strip(data, x=column, title=f"Strip Plot: {column}")
                        st.plotly_chart(fig_strip)
                    elif plot_choice == "Ridgeline Plot":
                        if len(data[column].dropna()) >= 10:
                            plt.figure(figsize=(8, 4))
                            sns.set(style="whitegrid")
                            sns.kdeplot(data=data[column].dropna(), shade=True, color="skyblue", label=column)
                            plt.title(f"Ridgeline Plot: {column}")
                            plt.xlabel(column)
                            plt.ylabel("Density")
                            st.pyplot(plt)
                    elif plot_choice == "Joy Plot":
                        if len(data[column].dropna()) >= 10:
                            plt.figure(figsize=(8, 4))
                            joy_data = [data[column].dropna()]
                            labels = [column]
                            fig, axes = joypy.joyplot(joy_data, labels=labels, title=f"Joy Plot: {column}")
                            st.pyplot(fig)
                    elif plot_choice == "Cumulative Distribution Plot":
                        fig_cdf = px.histogram(data, x=column, cumulative=True, title=f"Cumulative Distribution Plot: {column}")
                        st.plotly_chart(fig_cdf)
                    elif plot_choice == "Q-Q Plot":
                        plt.figure(figsize=(8, 4))
                        sm.qqplot(data[column].dropna(), line='s')
                        plt.title(f"Q-Q Plot: {column}")
                        st.pyplot()
                    elif plot_choice == "Probability Plot":
                        plt.figure(figsize=(8, 4))
                        stats.probplot(data[column].dropna(), dist="norm", plot=plt)
                        plt.title(f"Probability Plot: {column}")
                        st.pyplot()
                        
                    # Allow users to download the plot in a specific format
                    download_options = ["PNG", "JPEG", "PDF", "HTML"]
                    download_choice = st.sidebar.selectbox("Select Download Format", download_options)

                    if st.button("Download Plot"):
                        if download_choice == "PNG":
                            fig_hist.write_image(f"{column}_plot.png", format="png")
                        elif download_choice == "JPEG":
                            fig_hist.write_image(f"{column}_plot.jpeg", format="jpeg")
                        elif download_choice == "PDF":
                            fig_hist.write_image(f"{column}_plot.pdf", format="pdf")
                        elif download_choice == "HTML":
                            fig_hist.write_html(f"{column}_plot.html")

        # bi-variate plots
        if "Bivariate" in plot_categories:
            st.sidebar.subheader("Bivariate Plots")
            columns_to_plot = st.sidebar.multiselect("Select 2 columns for Bivariate Plots", data.columns)
            if len(columns_to_plot) == 2:
                col1, col2 = columns_to_plot

                if data[col1].dtype in [np.float64, np.int64] and data[col2].dtype in [np.float64, np.int64]:
                    # Allow users to choose which plot to display
                    plot_options = ["Scatter Plot", "Hexbin Plot", "Bubble Plot", "Heatmap", "Contour Plot", "Paired Density Plot", "Parallel Coordinates", "RadViz", "Andrews Plot", "Joint Plot", "Joint Plot (Hex)"]
                    plot_choice = st.sidebar.selectbox("Select a plot to display", plot_options)

                    # Plot the selected plot
                    if plot_choice == "Scatter Plot":
                        fig_scatter = px.scatter(data, x=col1, y=col2, title=f"Scatter Plot: {col1} vs {col2}")
                        st.plotly_chart(fig_scatter)
                    elif plot_choice == "Hexbin Plot":
                        fig_hexbin = px.density_heatmap(data, x=col1, y=col2, marginal_x='histogram', marginal_y='histogram',
                                                        title=f"Hexbin Plot: {col1} vs {col2}")
                        st.plotly_chart(fig_hexbin)
                    elif plot_choice == "Bubble Plot":
                        fig_bubble = px.scatter(data, x=col1, y=col2, size=col2, title=f"Bubble Plot: {col1} vs {col2}")
                        st.plotly_chart(fig_bubble)
                    elif plot_choice == "Heatmap":
                        fig_heatmap = px.imshow(data.corr(), x=columns_to_plot, y=columns_to_plot,
                                                color_continuous_scale='Blues', title=f"Heatmap: {col1} vs {col2}")
                        st.plotly_chart(fig_heatmap)
                    elif plot_choice == "Contour Plot":
                        fig_contour = px.density_contour(data, x=col1, y=col2, title=f"Contour Plot: {col1} vs {col2}")
                        st.plotly_chart(fig_contour)
                    elif plot_choice == "Paired Density Plot":
                        fig_paired_density = px.density_contour(data, x=col1, y=col2, marginal_x="histogram", marginal_y="histogram",
                                                                color_continuous_scale='Blues',
                                                                title=f"Paired Density Plot: {col1} vs {col2}")
                        st.plotly_chart(fig_paired_density)
                    elif plot_choice == "Parallel Coordinates":
                        fig_parallel_coordinates = px.parallel_coordinates(data, color=col2,
                                                                        title=f"Parallel Coordinates: {col1} vs {col2}")
                        st.plotly_chart(fig_parallel_coordinates)
                    elif plot_choice == "RadViz":
                        fig_radviz = px.radviz(data, color=col2, title=f"RadViz: {col1} vs {col2}")
                        st.plotly_chart(fig_radviz)
                    elif plot_choice == "Andrews Plot":
                        fig_andrews_plot = px.andrews_curves(data, color=col2, title=f"Andrews Plot: {col1} vs {col2}")
                        st.plotly_chart(fig_andrews_plot)
                    elif plot_choice == "Joint Plot":
                        sns.set(style="white")
                        sns.jointplot(x=col1, y=col2, data=data, kind="reg", height=8)
                        st.pyplot(plt)
                    elif plot_choice == "Joint Plot (Hex)":
                        sns.set(style="white")
                        sns.jointplot(x=col1, y=col2, data=data, kind="hex", height=8)
                        st.pyplot(plt)

                    # Allow users to download the plot in a specific format
                    download_options = ["PNG", "JPEG", "PDF", "HTML"]
                    download_choice = st.sidebar.selectbox("Select a format to download the plot", download_options)

                    if download_choice == "PNG":
                        fig = px.scatter(data, x=col1, y=col2, title=f"{plot_choice}: {col1} vs {col2}")
                        img_bytes = fig.to_image(format="png")
                        st.download_button(label="Download PNG", data=img_bytes, file_name=f"{plot_choice}_{col1}_vs_{col2}.png")
                    elif download_choice == "JPEG":
                        fig = px.scatter(data, x=col1, y=col2, title=f"{plot_choice}: {col1} vs {col2}")
                        img_bytes = fig.to_image(format="jpeg")
                        st.download_button(label="Download JPEG", data=img_bytes, file_name=f"{plot_choice}_{col1}_vs_{col2}.jpeg")
                    elif download_choice == "PDF":
                        fig = px.scatter(data, x=col1, y=col2, title=f"{plot_choice}: {col1} vs {col2}")
                        img_bytes = fig.to_image(format="pdf")
                        st.download_button(label="Download PDF", data=img_bytes, file_name=f"{plot_choice}_{col1}_vs_{col2}.pdf")
                    elif download_choice == "HTML":
                        fig = px.scatter(data, x=col1, y=col2, title=f"{plot_choice}: {col1} vs {col2}")
                        pio.write_html(fig, f"{plot_choice}_{col1}_vs_{col2}.html")
                        st.download_button(label="Download HTML", data=f"{plot_choice}_{col1}_vs_{col2}.html", file_name=f"{plot_choice}_{col1}_vs_{col2}.html")

        if "Multivariate" in plot_categories:
            st.sidebar.subheader("Multivariate Plots")                        
            columns_to_plot = st.sidebar.multiselect("Select at least 3 columns for Multivariate Plots", data.columns)
            if len(columns_to_plot) >= 3:
                # Allow users to choose which plot to display
                plot_options = ["Scatter Plot Matrix", "3D Scatter Plot", "Parallel Coordinates", "Correlation Heatmap", "Pairwise Scatter Plot", "Cluster Heatmap", "MDS Plot", "PCA Plot", "t-SNE Plot", "RadViz", "Network Plot", "Ternary Plot"]
                plot_choice = st.sidebar.selectbox("Select a plot to display", plot_options)

                # Plot the selected plot
                if plot_choice == "Scatter Plot Matrix":
                    fig_scatter_matrix = px.scatter_matrix(data[columns_to_plot])
                    st.plotly_chart(fig_scatter_matrix)
                elif plot_choice == "3D Scatter Plot":
                    fig_3d_scatter = px.scatter_3d(data, x=columns_to_plot[0], y=columns_to_plot[1], z=columns_to_plot[2], title=f"3D Scatter Plot: {columns_to_plot[0]} vs Others")
                    st.plotly_chart(fig_3d_scatter)
                elif plot_choice == "Parallel Coordinates":
                    fig_parallel_coordinates = px.parallel_coordinates(data, dimensions=columns_to_plot, color=columns_to_plot[0], title="Parallel Coordinates")
                    st.plotly_chart(fig_parallel_coordinates)
                elif plot_choice == "Correlation Heatmap":
                    corr_matrix = data[columns_to_plot].corr()
                    fig_corr_heatmap = px.imshow(corr_matrix, x=columns_to_plot, y=columns_to_plot,
                                                color_continuous_scale='Blues', title="Correlation Heatmap")
                    st.plotly_chart(fig_corr_heatmap)
                elif plot_choice == "Pairwise Scatter Plot":
                    fig_pairwise_scatter = px.scatter_matrix(data[columns_to_plot], title="Pairwise Scatter Plot")
                    st.plotly_chart(fig_pairwise_scatter)
                elif plot_choice == "Cluster Heatmap":
                    sns.set(style="whitegrid")
                    g = sns.clustermap(data[columns_to_plot].corr(), cmap="coolwarm", annot=True)
                    st.write("Cluster Heatmap")
                    st.pyplot(g)
                elif plot_choice == "MDS Plot":
                    from sklearn.manifold import MDS
                    mds = MDS(n_components=2)  # Specify the number of components as needed
                    mds_data = mds.fit_transform(data[columns_to_plot])
                    fig_mds = px.scatter(mds_data, x=0, y=1, title="MDS Plot")
                    st.plotly_chart(fig_mds)
                elif plot_choice == "PCA Plot":
                    from sklearn.decomposition import PCA
                    pca = PCA(n_components=3)  # Specify the number of components as needed
                    pca_data = pca.fit_transform(data[columns_to_plot])
                    fig_pca = px.scatter_3d(pca_data, x=0, y=1, z=2, title="PCA Plot")
                    st.plotly_chart(fig_pca)
                elif plot_choice == "t-SNE Plot":
                    from sklearn.manifold import TSNE
                    tsne = TSNE(n_components=2)  # Specify the number of components as needed
                    tsne_data = tsne.fit_transform(data[columns_to_plot])
                    fig_tsne = px.scatter(tsne_data, x=0, y=1, title="t-SNE Plot")
                    st.plotly_chart(fig_tsne)
                elif plot_choice == "RadViz":
                    plt.figure(figsize=(8, 4))
                    # Prepare the data by excluding the class_column (columns_to_plot[0])
                    data_for_radviz = data.drop(columns=[columns_to_plot[0]])
                    pd.plotting.radviz(data_for_radviz, columns_to_plot[0], ax=plt.gca())
                    plt.title(f"RadViz Plot: {columns_to_plot[0]} vs Others")
                    st.pyplot(plt)
                elif plot_choice == "Network Plot":
                    import networkx as nx
                    G = nx.random_geometric_graph(200, 0.125)
                    pos = nx.get_node_attributes(G, "pos")
                    dmin = 1
                    ncenter = 0
                    for n in pos:
                        x, y = pos[n]
                        d = (x - 0.5) ** 2 + (y - 0.5) ** 2
                        if d < dmin:
                            ncenter = n
                            dmin = d
                    p = dict(nx.single_source_shortest_path_length(G, ncenter))
                    fig_network = px.scatter(data, x=columns_to_plot[0], y=columns_to_plot[1])
                    st.plotly_chart(fig_network)
                elif plot_choice == "Ternary Plot":
                    import ternary
                    figure, tax = ternary.figure(scale=1.0)
                    tax.boundary(linewidth=2.0)
                    tax.gridlines(multiple=0.1, color="blue")
                    tax.set_title("Ternary Plot")
                    tax.scatter(data[columns_to_plot].values, marker='s', color='red', label="Data")
                    tax.legend()
                    tax.ticks(axis='lbr', linewidth=1, multiple=0.1)
                    st.pyplot(figure)

                # Allow users to download the plot in a specific format
                download_options = ["PNG", "JPEG", "PDF", "HTML"]
                download_choice = st.sidebar.selectbox("Select Download Format", download_options)

                if st.button("Download Plot"):
                    if download_choice == "PNG":
                        fig_hist.write_image(f"{column}_plot.png", format="png")
                    elif download_choice == "JPEG":
                        fig_hist.write_image(f"{column}_plot.jpeg", format="jpeg")
                    elif download_choice == "PDF":
                        fig_hist.write_image(f"{column}_plot.pdf", format="pdf")
                    elif download_choice == "HTML":
                        fig_hist.write_html(f"{column}_plot.html")
        
        # ML Task Selection
        st.sidebar.subheader("Machine Learning Tasks")
        X_columns = st.sidebar.multiselect("Select feature columns (X)", data.columns)
        y_column = st.sidebar.selectbox("Select target column (y)", data.columns)

        # Train-test split ratio
        split_ratio = st.sidebar.slider("Select train-test split ratio (%)", 10, 90, 80)
        st.sidebar.text(f"Train set size: {split_ratio}%")
        st.sidebar.text(f"Test set size: {100-split_ratio}%")

        if y_column in data.columns:
                task_type = st.sidebar.radio("Select Task Type", ["Regression", "Classification"])
                X = data[X_columns]
                y = data[y_column]

                # Scaling
                scaler = StandardScaler()
                X = scaler.fit_transform(X)

                # Splitting data
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=(100 - split_ratio) / 100.0, random_state=42)

                if task_type == "Regression":
                    st.sidebar.warning("This is a Regression Problem!")
                    trained_regression_models = {}

                    # Regression models selection
                    regression_models = {
                    "Linear Regression": LinearRegression(),
                    "Ridge Regression": Ridge(),
                    "Lasso Regression": Lasso(), 
                    "Elastic Net": ElasticNet(),
                    "SVR": SVR(),
                    "Decision Tree": DecisionTreeRegressor(),
                    "Random Forest": RandomForestRegressor(),
                    "AdaBoost": AdaBoostRegressor(), 
                    "Gradient Boosting": GradientBoostingRegressor(),
                    "XGBoost": XGBRegressor(),
                    "LightGBM": LGBMRegressor(),  
                    "Bayesian Linear": BayesianRidge(),
                    "Bayesian Ridge": BayesianRidge(),
                    "Poisson Regression": PoissonRegressor(),
                    "Logistic Regression": LogisticRegression(),
                    "MLP": MLPRegressor(),
                    "kNN": KNeighborsRegressor(),
                    "ARIMA": ARIMA(),
                    "Exponential Smoothing": ExponentialSmoothing(),
                    "Prophet": Prophet(),
                    "Gaussian Process": GaussianProcessRegressor(),
                    "Isotonic": IsotonicRegression(),
                    "Robust": RANSACRegressor(),
                    "CART": DecisionTreeRegressor(),
                    "Regression Tree": DecisionTreeRegressor()  
                    }

                    selected_regression_models = st.sidebar.multiselect("Select Regression Models", list(regression_models.keys()))

                    if selected_regression_models:
                        regression_results = {}
                        for model_name in selected_regression_models:
                            model = regression_models[model_name]
                            model.fit(X_train, y_train)
                            trained_regression_models[model_name] = model

                    if selected_regression_models:
                        regression_results = {}
                        for model_name in selected_regression_models:
                            model = regression_models[model_name]
                            model.fit(X_train, y_train)
                            y_pred = model.predict(X_test)
                            mse = mean_squared_error(y_test, y_pred)
                            mae = mean_absolute_error(y_test, y_pred)
                            r2 = r2_score(y_test, y_pred)
                            rmse = np.sqrt(mse)
                            evs = explained_variance_score(y_test, y_pred)
                            regression_results[model_name] = [mse, mae, r2, rmse, evs]

                        st.write("### Regression Model Evaluation:")
                        selected_metrics = st.multiselect("Select metrics to compare", ["Mean Squared Error", "Mean Absolute Error", "R-squared", "Root Mean Squared Error", "Explained Variance Score"])

                        for metric in selected_metrics:
                            best_regression_model_name = max(regression_results, key=lambda k: regression_results[k][selected_metrics.index(metric)])
                            st.write(f"Best {metric} Model: {best_regression_model_name}")
                            st.write(f"{metric}: {regression_results[best_regression_model_name][selected_metrics.index(metric)]}")

                        st.write("### Best Regression Model:")
                        st.write(regression_results)

                        save_regression_model_name = st.button("Save Best Regression Model")
                        if save_regression_model_name:
                            best_regression_model = trained_regression_models[best_regression_model_name]
                            with open("best_regression_model.pkl", "wb") as model_file:
                                pickle.dump(best_regression_model, model_file)
                            st.write("Best regression model saved as 'best_regression_model.pkl'")
                elif task_type == "Classification":
                    st.sidebar.warning("This is a Classification Problem!")

                    # Encoding the target column if it's categorical
                    le = LabelEncoder()
                    y = le.fit_transform(y)

                    # Classification models selection
                    classification_models = {
                        "Logistic Regression": LogisticRegression(),
                        "Random Forest Classifier": RandomForestClassifier(),
                        "Gradient Boosting Classifier": GradientBoostingClassifier(),
                        "Support Vector Classifier": SVC(),
                        "K-Nearest Neighbors": KNeighborsClassifier(),
                        "Decision Tree Classifier": DecisionTreeClassifier(),
                        "Gaussian Naive Bayes": GaussianNB(),
                        "Multinomial Naive Bayes": MultinomialNB(),
                        "Multi-Layer Perceptron": MLPClassifier(),
                        "Quadratic Discriminant Analysis": QuadraticDiscriminantAnalysis(),
                        "Linear Discriminant Analysis": LinearDiscriminantAnalysis(),
                        "Perceptron": Perceptron(),
                        "Passive Aggressive Classifier": PassiveAggressiveClassifier(),
                        "SGD Classifier": SGDClassifier(),
                        "Radius Neighbors Classifier": RadiusNeighborsClassifier(),
                        "Nearest Centroid Classifier": NearestCentroid(),
                        "Gaussian Process Classifier": GaussianProcessClassifier(),
                        "Nu-Support Vector Classifier": NuSVC(),
                        "Label Propagation": LabelPropagation(),
                        "Label Spreading": LabelSpreading(),
                        "Extra Trees Classifier": ExtraTreesClassifier()
                        }

                    selected_classification_models = st.sidebar.multiselect("Select Classification Models", list(classification_models.keys()))

                    if selected_classification_models:
                        classification_results = {}
                        for model_name in selected_classification_models:
                            model = classification_models[model_name]
                            model.fit(X_train, y_train)
                            y_pred = model.predict(X_test)
                            accuracy = accuracy_score(y_test, y_pred)
                            precision = precision_score(y_test, y_pred)
                            recall = recall_score(y_test, y_pred)
                            f1 = f1_score(y_test, y_pred)
                            roc_auc = roc_auc_score(y_test, y_pred)
                            classification_results[model_name] = [accuracy, precision, recall, f1, roc_auc]

                        st.write("### Classification Model Evaluation:")
                        selected_metrics = st.multiselect("Select metrics to compare", ["Accuracy", "Precision", "Recall", "F1-Score", "ROC AUC"])

                        for metric in selected_metrics:
                            best_classification_model_name = max(classification_results, key=lambda k: classification_results[k][selected_metrics.index(metric)])
                            st.write(f"Best {metric} Model: {best_classification_model_name}")
                            st.write(f"{metric}: {classification_results[best_classification_model_name][selected_metrics.index(metric)]}")

                        st.write("### Best Classification Model:")
                        st.write(classification_results)

                        save_classification_model_name = st.button("Save Best Classification Model")
                        if save_classification_model_name:
                            best_classification_model = classification_models[best_classification_model_name]
                            with open("best_classification_model.pkl", "wb") as model_file:
                                pickle.dump(best_classification_model, model_file)
                            st.write("Best classification model saved as 'best_classification_model.pkl'")

                        # Save Best Regression Model
                        save_regression_model_name = st.button("Save Best Regression Model")
                        if save_regression_model_name:
                            best_regression_model = trained_regression_models[best_regression_model_name]
                            with open("best_regression_model.pkl", "wb") as model_file:
                                pickle.dump(best_regression_model, model_file)
                            st.write("Best regression model saved as 'best_regression_model.pkl'")

                        # Upload non-labeled data and get predictions
        if st.sidebar.checkbox("Upload non-labeled data and get predictions"):
            st.sidebar.subheader("Upload Non-Labeled Data")
            uploaded_test_file = st.sidebar.file_uploader("Upload your non-labeled dataset (CSV)", type=["csv"])
            if uploaded_test_file:
                test_data = pd.read_csv(uploaded_test_file)
                st.write("### Sneak peek of the non-labeled data:")
                st.write(test_data.head())

                test_numeric_data = test_data.select_dtypes(include=[np.number])
                test_data[test_numeric_data.columns] = imputer.transform(test_data[test_numeric_data.columns])

                if encoding_method == "Label Encoding":
                    categorical_cols = []  # Define the categorical columns here
                    for col in categorical_cols:
                        test_data[col] = label_encoder.transform(test_data[col])
                elif encoding_method == "One-Hot Encoding":
                    test_data = pd.get_dummies(test_data, columns=categorical_cols)

                if task_type == "Regression":
                    best_regression_model = None
                    with open("best_regression_model.pkl", "rb") as model_file:
                        best_regression_model = pickle.load(model_file)

                    if best_regression_model:
                        st.write("### Predictions using the Best Regression Model:")
                        test_predictions = best_regression_model.predict(test_data[X_columns])
                        st.write(test_predictions)

                elif task_type == "Classification":
                    best_classification_model = None
                    with open("best_classification_model.pkl", "rb") as model_file:
                        best_classification_model = pickle.load(model_file)

                    if best_classification_model:
                        st.write("### Predictions using the Best Classification Model:")
                        test_predictions = best_classification_model.predict(test_data[X_columns])
                        st.write(test_predictions)

if __name__ == "__main__":
    main()

import streamlit as st
from PIL import Image
import numpy as np
from config import *
from tensorflow.keras.models import load_model # type: ignore
from sklearn.metrics import confusion_matrix
import plotly.express as px
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import time
from datetime import datetime


# Load the trained model
model = load_model(MODEL_PATH)


def predict_page():
    """Detection Page"""
    st.title("Detection of Alzheimer's Disease")
    st.markdown("This application uses a pre-trained model to Detect Alzheimer's disease from MRI scans.")
    uploaded_file = st.file_uploader("Upload an MRI scan to check for Alzheimer's disease.", type=["jpg", "jpeg", "png"])

    

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
    
        predicted_idx = predict(image)
        class_labels = ['Mild Demented', 'Moderate Demented', 'Non Demented', 'Very Mild Demented']
        class_sentence = [
    """🧬 **Common Symptoms of Mild Demented Stage:**
        At this stage, memory and cognitive issues become more noticeable. Individuals may struggle with complex tasks like managing finances or planning events. 
        They may repeat questions, get lost in familiar places, or show difficulty recalling recent conversations, though they can still manage basic self-care.
        - Forgetting recent conversations or events  
        - Struggling with decision-making  
        - Losing track of time or place  
        - Difficulty managing money or organizing tasks  
        - Subtle personality or mood changes""",

        """🧬 **Moderate Demented Stage:**
        The individual requires more support with daily activities. Memory loss is more pronounced, and confusion is common, 
        even about personal history or surroundings. They may need help with dressing, hygiene, and cooking. 
        Behavioral changes, such as agitation or suspicion, can begin to emerge.
        - Increased memory loss  
        - Frequent confusion  
        - Help required with daily activities  
        - Noticeable personality changes""",

            """🧠 **Non Demented (Healthy):**
        The individual exhibits no significant cognitive impairment. Daily activities are performed independently, 
        memory and reasoning are within normal limits, and there are no observable signs of dementia. 
        The person remains socially and professionally active without any assistance.
        - No signs of cognitive impairment detected  
        - Maintain a healthy lifestyle  
        - Stay mentally active  
        - Engage in physical and social activities regularly  
        - Support long-term brain health""",

            """🧬 **Very Mild Demented Stage:**
        The person may begin to show very slight memory lapses, such as forgetting names or misplacing objects. 
        However, these symptoms are subtle and often mistaken for normal aging. There is no noticeable impact on daily functioning or independence.
        - Occasional memory lapses  
        - Slight trouble with focus  
        - Can still function independently"""]
        
    
        predicted_label = class_labels[predicted_idx]
        predicted_sentence = class_sentence[predicted_idx]


        tab1, tab2, tab3 = st.tabs(["📈 Model Training", "📈 Models Metrics","📋 Detection Result"])


        time.sleep(2)
        with tab1:   # Simulate a delay for classification

            
            st.subheader("📁 Reading Data From Dataset")

            dataset_info = """
            Found 33984 files belonging to 4 classes.
            Using 27188 files for training.
            Found 33984 files belonging to 4 classes.
            Using 6796 files for validation.
            """

            st.code(dataset_info, language="text")

            time.sleep(1)
            st.subheader("Splitting validation and testing datasets")

            st.markdown("As we see, the number of full_validation_dataset batches "
            "is **213 batch** and we want to split it into validation and " \
            "testing datasets of size **106,107 batch**.")

            batch_sizeinfo = """
            Total number of full_validation_dataset batches : 213
            Number of batches in validation dataset : 106
            Number of batches in test dataset : 107
            """
            st.code(batch_sizeinfo, language="text")

            time.sleep(1)
            st.subheader("Classes names")

            class_info = """
            ['Mild_Demented', 'Moderate_Demented', 'Non_Demented', 'Very_Mild_Demented']
            """
            st.code(class_info, language="text")


            time.sleep(2)
            st.subheader("📸 Before and After Data Augmentation")
            st.markdown("These images show how data augmentation techniques like rotation, " \
            "flipping, zooming, etc., are applied to improve generalization "
            "and increase dataset diversity.")

                        # Load uploaded screenshots
            before_aug = Image.open(BEFORE_AUG)
            
            after_aug = Image.open(AFTER_AUG)

            st.image(before_aug, caption="Before Augmentation - Moderate Demented Samples")

                        
            st.image(after_aug, caption="After Augmentation - Transformed Samples")

            


            time.sleep(1)
            st.subheader("Classes weights")

            weight_info = """
            Weight for class "MildDemented" : 0.95
            Weight for class "ModerateDemented" : 1.31
            Weight for class "NonDemented" : 0.89
            Weight for class "VeryMildDemented" : 0.95
            """
            st.code(weight_info, language="text")

            # Simulated class-wise performance data
            data = {
                    "Class": ["Mild Demented", "Moderate Demented", "Non Demented", "Very Mild Demented"],
                    "Weight": [0.95, 1.31, 0.89, 0.95],
                }
            df = pd.DataFrame(data)

            # Bar chart for Precision
            # st.subheader("🎯 Weight per Class")
            # fig_precision = px.bar(df, x="Class", y="Weight", color="Class", text="Weight", range_y=[0, 1.5])
            # st.plotly_chart(fig_precision, use_container_width=True)





            # st.subheader("Model Summary")



            # model_summary = """
            #             Model: "EfficientNetB0"
            #             ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
            #             ┃ Layer (type)                      ┃ Output Shape            ┃         Param # ┃
            #             ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
            #             │ input_layer_3 (InputLayer)        │ (None, 128, 128, 3)     │               0 │
            #             ├───────────────────────────────────┼─────────────────────────┼─────────────────┤
            #             │ data_augmentation (Sequential)    │ (None, 128, 128, 3)     │               0 │                               
            #             ├───────────────────────────────────┼─────────────────────────┼─────────────────┤
            #             │ efficientnetb0 (Functional)       │ (None, 4, 4, 1280)      │       4,049,571 │
            #             ├───────────────────────────────────┼─────────────────────────┼─────────────────┤
            #             │ global_average_pooling2d          │ (None, 1280)            │               0 │
            #             │ (GlobalAveragePooling2D)          │                         │                 │
            #             ├───────────────────────────────────┼─────────────────────────┼─────────────────┤
            #             │ dense (Dense)                     │ (None, 4)               │           5,124 │
            #             └───────────────────────────────────┴─────────────────────────┴─────────────────┘
            #             Total params: 4,054,695 (15.47 MB)
            #             Trainable params: 1,356,084 (5.17 MB)
            #             Non-trainable params: 2,698,611 (10.29 MB)
            #             """

            # st.code(model_summary, language="text")
            # st.markdown("The model summary above shows the architecture of the EfficientNetB0 model used for Alzheimer's detection. It includes the input layer, data augmentation, and the final dense layer with 4 output classes.")            
                        

            time.sleep(1)
            st.subheader("📈 EfficientNetB0 Training Dashboard")
            st.markdown("Visual summary of **Train/Validation Accuracy & Loss** over 50 Epochs + final **Confusion Matrix**.")

                        # --- Static Training Data (from your CSV) ---
            data1 = {
                            "Epoch": list(range(1, 51)),
                            "Train Accuracy": [
                                0.7244, 0.7717, 0.8027, 0.8287, 0.8507, 0.8699, 0.8798, 0.8963, 0.9042, 0.9103,
                                0.9187, 0.9263, 0.9305, 0.9355, 0.9446, 0.9388, 0.9399, 0.9445, 0.9503, 0.9510,
                                0.9519, 0.9529, 0.9555, 0.9561, 0.9569, 0.9635, 0.9591, 0.9604, 0.9636, 0.9643,
                                0.9687, 0.9647, 0.9663, 0.9677, 0.9702, 0.9674, 0.9679, 0.9711, 0.9749, 0.9718,
                                0.9758, 0.9720, 0.9726, 0.9743, 0.9737, 0.9746, 0.9757, 0.9740, 0.9753, 0.9735
                                ],
                            "Validation Accuracy": [
                                0.7462, 0.8187, 0.8213, 0.8491, 0.8685, 0.8626, 0.898, 0.9059, 0.9139, 0.9313, 
                                0.913, 0.9287, 0.9399, 0.9422, 0.9404, 0.9337, 0.9469, 0.9458, 0.9543, 0.9552, 
                                0.9546, 0.9553, 0.9614, 0.9649, 0.9571, 0.9676, 0.9584, 0.9588, 0.9584, 0.9676, 
                                0.9785, 0.9717, 0.962, 0.9717, 0.9767, 0.9702, 0.9732, 0.9696, 0.9746, 0.9746, 
                                0.9802, 0.9744, 0.9791, 0.967, 0.9755,0.9823, 0.9767, 0.9797, 0.9773, 0.9702
                            ],
                            "Train Loss": [
                                0.6558, 0.5348, 0.4789, 0.4243, 0.3752, 0.3346, 0.3061, 0.2731, 0.252, 
                                0.2325, 0.2117, 0.1994, 0.1866, 0.1746, 0.1583, 0.1633, 0.1599, 0.1541, 
                                0.1424, 0.1417, 0.1396, 0.1362, 0.1313, 0.1293, 0.1274, 0.1171, 0.1223, 
                                0.1195, 0.1166, 0.1147, 0.1039, 0.1105, 0.1085, 0.1068, 0.1009, 0.1062, 
                                0.1033, 0.0975, 0.0911, 0.0976, 0.0881, 0.0955, 0.0934, 0.0917, 0.0919, 
                                0.0886, 0.0883, 0.0903, 0.089, 0.0924
                            ],
                            "Validation Loss": [
                                0.5594, 0.4492, 0.4357, 0.3809, 0.3424, 0.3409, 0.2882, 0.2646, 0.2443, 
                                0.2132, 0.2312, 0.2197, 0.1932, 0.1873, 0.1864, 0.1857, 0.1751, 0.1755, 
                                0.1537, 0.1519, 0.1504, 0.1492, 0.1398, 0.1309, 0.1451, 0.1226, 0.1418, 
                                0.1408, 0.1413, 0.1232, 0.0963, 0.1118, 0.1332, 0.1095, 0.0984, 0.1098, 
                                0.1042, 0.1103, 0.1033, 0.1026, 0.0903, 0.103, 0.0934, 0.1186, 0.1009, 
                                0.0844, 0.0991, 0.0945, 0.0973, 0.1102
                            ]
                        }   


            df = pd.DataFrame(data1)

                        # --- Accuracy Plots ---
            

            st.subheader("📉 Training vs Validation Accuracy")

            fig = px.line(df, x="Epoch", y=["Train Accuracy", "Validation Accuracy"], markers=True)
            fig.update_layout(
                title="Train and Validation Accuracy over Epochs",
                xaxis_title="Epoch",
                yaxis_title="Accuracy",
                legend_title="Accuracy Type"
            )
            st.plotly_chart(fig, use_container_width=True)

                        
            st.subheader("📉 Training vs Validation Loss")

            fig = px.line(df, x="Epoch", y=["Train Loss", "Validation Loss"], markers=True)
            fig.update_layout(
                title="Train and Validation Loss over Epochs",
                xaxis_title="Epoch",
                yaxis_title="Loss",
                legend_title="Loss Type"
            )
            st.plotly_chart(fig, use_container_width=True)






            time.sleep(2)
            st.title("Confusion Matrix Visualization")

                        # Labels
            labels = ["Mild Demented", "Moderate Demented", "Non Demented", "Very Mild Demented"]


                        # Confusion matrix values from your image
            conf_matrix = np.array([
                            [912, 0, 9, 3],
                            [0, 607, 0, 0],
                            [2, 0, 976, 10],
                            [1, 0, 7, 806]
                        ])

                        # Plotting function
            def plot_confusion_matrix(cm, labels):
                            fig, ax = plt.subplots(figsize=(9, 5))
                            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                                        xticklabels=labels, yticklabels=labels, ax=ax)
                            ax.set_xlabel("Predicted label")
                            ax.set_ylabel("True label")
                            ax.set_title("Confusion Matrix")
                            return fig

                        # Display the matrix
            fig = plot_confusion_matrix(conf_matrix, labels)
            st.pyplot(fig)
            st.markdown("A confusion matrix is a table used to evaluate the performance of a classification model by comparing predicted and actual labels. " \
            "It displays true positives, true negatives, false positives, and false negatives to give insight into model accuracy and errors.")


        time.sleep(2)
        with tab2:
                st.subheader("📈 Model Performance Metrics")
                model_comparisons = {
                                    "Model": ["EfficientNet-B0", "ResNet50", "VGG16", "MobileNetV2"],
                                    "Accuracy (%)": [97, 92, 88, 57],
                                    "AUC": [0.9983, 0.8271, 0.9933, 0],
                                    "Precision": [0.9705, 0.8762, 0.9225, 0.6447],
                                    "Recall": [0.9743, 0.8748, 0.9127, 0.7955]
                                }
                df = pd.DataFrame(model_comparisons)
                df["🏆"] = df["Accuracy (%)"].apply(lambda x: "✅" if x == max(df["Accuracy (%)"]) else "")

                st.subheader("📊 Interactive Table")
                st.dataframe(df, use_container_width=True)
                
                # ---- MODEL SELECTION TOGGLE ----
                st.subheader("📈 Epoch-wise Accuracy Comparison")

                # Allow model selection
                available_models1 = ["EfficientNet-B0", "ResNet50", "VGG16", "MobileNetV2"]
                selected_models1 = st.multiselect("Select models to compare:", available_models1, default=available_models1)
                epochs = list(range(1, 51))

                epoch_data1 = {
                    "Epoch": epochs * 4,
                    "Accuracy": 
                        [ 0.7462, 0.8187, 0.8213, 0.8491, 0.8685, 0.8626, 0.898, 0.9059, 0.9139, 0.9313, 
                        0.913, 0.9287, 0.9399, 0.9422, 0.9404, 0.9337, 0.9469, 0.9458, 0.9543, 0.9552, 
                        0.9546, 0.9553, 0.9614, 0.9649, 0.9571, 0.9676, 0.9584, 0.9588, 0.9584, 0.9676, 
                        0.9785, 0.9717, 0.962, 0.9717, 0.9767, 0.9702, 0.9732, 0.9696, 0.9746, 0.9746, 
                        0.9802, 0.9744, 0.9791, 0.967, 0.9755,0.9823, 0.9767, 0.9797, 0.9773, 0.9702] +  # EfficientNet-B0
                        [0.5086, 0.4977, 0.5149, 0.5462, 0.6401, 0.6933, 0.6275, 0.6964, 0.7199, 0.7011, 
                         0.7512, 0.7778, 0.7621, 0.7590, 0.7731, 0.7840, 0.7966, 0.7780, 0.8310, 0.8435, 
                         0.7919, 0.8091, 0.8216, 0.7919, 0.8310, 0.8607, 0.8513, 0.8466, 0.8623, 0.8560,
                         0.8545, 0.8545, 0.8717, 0.8920, 0.8826, 0.9014, 0.9139, 0.8764, 0.8529, 0.8998, 
                         0.8889, 0.9139, 0.8873, 0.8889, 0.8654, 0.8967, 0.9067, 0.9014, 0.8732, 0.8748] +  # ResNet50
                        [0.5008, 0.5023, 0.6119, 0.7293, 0.7152, 0.7981, 0.8357, 0.7778, 0.8357, 0.8638, 
                         0.8889, 0.8826, 0.8920, 0.8701, 0.8858, 0.9092, 0.9186, 0.9124, 0.9171, 0.9186, 
                         0.9186, 0.9186, 0.9186, 0.9186, 0.9186, 0.9186, 0.9186, 0.9186, 0.9186, 0.9186,
                         0.9186, 0.9186, 0.9186, 0.9186, 0.9186, 0.9186, 0.9186, 0.9186, 0.9186, 0.9186,
                         0.9186, 0.9186, 0.9186, 0.9186, 0.9186, 0.9186, 0.9186, 0.9186, 0.9186, 0.9186] +  # VGG16
                         [0.5008, 0.5008, 0.3975, 0.4147, 0.5164, 0.5180, 0.5055, 0.5102, 0.5274, 0.5258, 
                          0.5336, 0.5634, 0.5211, 0.5039, 0.5070, 0.5274, 0.5649, 0.5775, 0.5665, 0.5587, 
                          0.5509, 0.5352, 0.5634, 0.5759, 0.5743, 0.5759, 0.5822, 0.5790, 0.5853, 0.5806, 
                          0.5618, 0.5837, 0.5243, 0.5196, 0.5415, 0.5931, 0.5775, 0.5837, 0.5900, 0.5696, 
                          0.5665, 0.5822, 0.5978, 0.5798, 0.5696, 0.5696, 0.5939, 0.6025, 0.6150, 0.5696],   # MobileNetV2
                    "Model": ["EfficientNet-B0"] * 50 + ["ResNet50"] * 50 + ["VGG16"] * 50 + ["MobileNetV2"] * 50
                }

                epoch_df1 = pd.DataFrame(epoch_data1)
                filtered_df1 = epoch_df1[epoch_df1["Model"].isin(selected_models1)]

                fig_line1 = px.line(
                    filtered_df1,
                    x="Epoch",
                    y="Accuracy",
                    color="Model",
                    markers=True,
                    title="Epoch-wise Accuracy Progression",
                    labels={"Accuracy": "Accuracy", "Epoch": "Epoch Number"}
                )
                fig_line1.update_layout(yaxis_range=[0.5, 1.0], height=500)
                st.plotly_chart(fig_line1, use_container_width=True)

                # ---- MODEL DATA TABLE ----
               

                # ---- AUC Chart ----
                st.subheader("📈 Epoch-wise AUC Comparison")
                available_models2 = ["EfficientNet-B0", "ResNet50", "VGG16"]
                selected_models2 = st.multiselect("Select models to compare (AUC):", available_models2, default=available_models2)

                epoch_data2 = {
                    "Epoch": epochs * 3,
                    "AUC": 
                        [0.8493, 0.9050, 0.9492, 0.9697, 0.9772, 0.9813, 0.9839, 0.9874, 0.9886, 0.9901,
                        0.9916, 0.9927, 0.9926, 0.9942, 0.9946, 0.9949, 0.9949, 0.9955, 0.9961, 0.9959,
                        0.9966, 0.9967, 0.9967, 0.9961, 0.9969, 0.9978, 0.9970, 0.9975, 0.9977, 0.9974,
                        0.9978, 0.9980, 0.9981, 0.9979, 0.9980, 0.9981, 0.9980, 0.9982, 0.9982, 0.9983,
                        0.9986, 0.9982, 0.9982, 0.9982, 0.9984, 0.9984, 0.9987, 0.9989, 0.9987, 0.9983
                        ] +  
                        [0.8283, 0.8195, 0.8326, 0.8490, 0.8740, 0.9022, 0.8943, 0.9207, 0.9298, 
                         0.9208, 0.9412, 0.9467, 0.9456, 0.9476, 0.9476, 0.9543, 0.9524, 0.9543, 
                         0.9627, 0.9750, 0.9528, 0.9675, 0.9467, 0.9646, 0.9723, 0.9760, 
                         0.9701, 0.9803, 0.9752, 0.9686, 0.9766, 0.9797, 0.9844, 0.9848, 0.9861, 
                         0.9870, 0.9776, 0.9758, 0.9862, 0.9836, 0.9800, 0.9758, 0.9785, 0.9658, 
                         0.9871, 0.9814, 0.9889, 0.9751, 0.9727] +  
                        [0.8271, 0.8347, 0.8684, 0.9202, 0.9214, 0.9565, 0.9697, 0.9500, 0.9675, 
                         0.9774, 0.9789, 0.9812, 0.9875, 0.9824, 0.9861, 0.9907, 0.9915, 0.9911, 0.9926, 
                         0.9933, 0.9933, 0.9933, 0.9933, 0.9933, 0.9933, 0.9933, 0.9933, 0.9933, 0.9933, 0.9933, 0.9933,
                         0.9933, 0.9933, 0.9933, 0.9933, 0.9933, 0.9933, 0.9933, 0.9933, 0.9933, 0.9933,
                         0.9933, 0.9933, 0.9933, 0.9933, 0.9933, 0.9933, 0.9933, 0.9933, 0.9933, 0.9933
                         ],  
                    "Model": ["EfficientNet-B0"] * 50 + ["ResNet50"] * 50 + ["VGG16"] * 50
                }
                epoch_df2 = pd.DataFrame(epoch_data2)
                filtered_df2 = epoch_df2[epoch_df2["Model"].isin(selected_models2)]

                fig_line2 = px.line(
                    filtered_df2,
                    x="Epoch",
                    y="AUC",
                    color="Model",
                    markers=True,
                    title="Epoch-wise AUC Progression",
                    labels={"AUC": "AUC", "Epoch": "Epoch Number"}
                )
                fig_line2.update_layout(yaxis_range=[0.5, 1.0], height=500)
                st.plotly_chart(fig_line2, use_container_width=True)

                # ---- Precision Chart ----
                st.subheader("📈 Epoch-wise Precision Comparison")
                selected_models3 = st.multiselect("Select models to compare (Precision):", available_models1, default=available_models1)

                epoch_data3 = {
                    "Epoch": epochs * 4,
                    "Precision": [
                            0.7875, 0.8290, 0.8419, 0.8615, 0.8763, 0.9037, 0.9034, 0.9129, 0.9166, 0.9328,
                            0.9170, 0.9321, 0.9414, 0.9446, 0.9378, 0.9404, 0.9480, 0.9410, 0.9560, 0.9554,
                            0.9554, 0.9554, 0.9639, 0.9654, 0.9592, 0.9687, 0.9598, 0.9524, 0.9595, 0.9696,
                            0.9791, 0.9731, 0.9622, 0.9731, 0.9784, 0.9716, 0.9735, 0.9705, 0.9749, 0.9758,
                            0.9808, 0.9762, 0.9799, 0.9675, 0.9758, 0.9823, 0.9776, 0.9799, 0.9773, 0.9705
                        ] +  
                        [0.8720, 0.5484, 0.7326, 0.7381, 0.8169, 0.8253, 0.6839, 0.7326, 0.7735, 0.7241, 
                         0.7742, 0.7976, 0.7789, 0.7840, 0.7855, 0.7987, 0.8139, 0.7861, 0.8438, 0.8539, 
                         0.8029, 0.8152, 0.8251, 0.7918, 0.8346, 0.8610, 0.8569, 0.8533, 0.8683, 0.8567, 
                         0.8562, 0.8578, 0.8768, 0.8915, 0.8880, 0.9113, 0.9197, 0.8796, 0.8538, 0.9013, 
                         0.8888, 0.9180, 0.8887, 0.8903, 0.8668, 0.8995, 0.9066, 0.9071, 0.8760, 0.8762] +  
                        [0.5426, 0.5327, 0.6886, 0.8444, 0.7450, 0.8278, 0.8610, 0.7993, 0.8479, 0.8700, 0.8919, 
                         0.8831, 0.9029, 0.8792, 0.8922, 0.9118, 0.9185, 0.9194, 0.9242, 0.9225, 0.9225, 0.9225, 
                         0.9225, 0.9225, 0.9225, 0.9225, 0.9225, 0.9225, 0.9225, 0.9225,
                         0.9225, 0.9225, 0.9225, 0.9225, 0.9225, 0.9225, 0.9225, 0.9225, 0.9225, 0.9225,
                         0.9225, 0.9225, 0.9225, 0.9225, 0.9225, 0.9225, 0.9225, 0.9225, 0.9225, 0.9225
                         ] +  
                        [0.5000, 0.5000, 0.4099, 0.4199, 0.5164, 0.5197, 0.5791, 0.5440, 0.5553, 
                         0.5737, 0.5945, 0.6789, 0.5385, 0.5381, 0.5966, 0.6055, 0.6721, 0.6564, 0.6164, 
                         0.6181, 0.5963, 0.6193, 0.6157, 0.6310, 0.6558, 0.6300, 0.6769, 0.6247, 0.6627, 
                         0.6631, 0.5934, 0.6392, 0.5583, 0.5551, 0.5966, 0.6927, 0.6647, 0.6088, 0.6340, 
                         0.6267, 0.6287, 0.6633, 0.6854, 0.6373, 0.6278, 0.6281, 0.6366, 0.7394, 0.7186, 0.6447],   
                    "Model": ["EfficientNet-B0"] * 50 + ["ResNet50"] * 50 + ["VGG16"] * 50 + ["MobileNetV2"] * 50
                }
                epoch_df3 = pd.DataFrame(epoch_data3)
                filtered_df3 = epoch_df3[epoch_df3["Model"].isin(selected_models3)]

                fig_line3 = px.line(
                    filtered_df3,
                    x="Epoch",
                    y="Precision",
                    color="Model",
                    markers=True,
                    title="Epoch-wise Precision Progression",
                    labels={"Precision": "Precision", "Epoch": "Epoch Number"}
                )
                fig_line3.update_layout(yaxis_range=[0.5, 1.0], height=500)
                st.plotly_chart(fig_line3, use_container_width=True)

                # ---- Recall Chart ----
                st.subheader("📈 Epoch-wise Recall Comparison")
                selected_models4 = st.multiselect("Select models to compare (Recall):", available_models1, default=available_models1)

                epoch_data4 = {
                    "Epoch": epochs * 4,
                    "Recall": 
                         [
                            0.7733, 0.8076, 0.8277, 0.8499, 0.8685, 0.8826, 0.8934, 0.9059, 0.9132, 0.9182,
                            0.9250, 0.9311, 0.9355, 0.9395, 0.9428, 0.9436, 0.9434, 0.9485, 0.9531, 0.9537,
                            0.9546, 0.9555, 0.9586, 0.9603, 0.9593, 0.9652, 0.9612, 0.9630, 0.9654, 0.9663,
                            0.9704, 0.9661, 0.9678, 0.9693, 0.9733, 0.9689, 0.9694, 0.9725, 0.9715, 0.9733,
                            0.9772, 0.9736, 0.9741, 0.9755, 0.9750, 0.9752, 0.9768, 0.9751, 0.9762, 0.9743
                        ] +  
                        [0.1518, 0.4695, 0.3302, 0.3396, 0.3772, 0.3772, 0.5383, 0.6432, 0.6573, 0.6573, 0.7246, 
                         0.7402, 0.7387, 0.7496, 0.7621, 0.7762, 0.7872, 0.7590, 0.8114, 0.8326, 0.7778, 0.8075,
                         0.8122, 0.7856, 0.8294, 0.8529, 0.8435, 0.8372, 0.8576, 0.8513, 0.8482, 0.8498, 
                         0.8498, 0.8773, 0.8811, 0.8990, 0.9139, 0.8685, 0.8498, 0.8990, 0.8804, 0.9108, 
                         0.8873, 0.8809, 0.8654, 0.8967, 0.9051, 0.9014, 0.8732, 0.8748] +  
                        [0.4789, 0.4977, 0.3772, 0.4757, 0.6448, 0.7371, 0.7950, 0.7606, 0.8200, 0.8592, 
                         0.8779, 0.8748, 0.8873, 0.8654, 0.8811, 0.9061, 0.9171, 0.9108, 0.9155, 0.9124,0.9124, 
                         0.9124, 0.9124, 0.9124, 0.9124, 0.9124, 0.9124, 0.9124, 0.9124, 0.9124,
                         0.9124, 0.9124, 0.9124, 0.9124, 0.9124, 0.9124, 0.9124, 0.9124, 0.9124, 0.9124,
                         0.9124, 0.9124, 0.9124, 0.9124, 0.9124, 0.9124, 0.9124, 0.9124, 0.9124, 0.9124
                        ] +  
                        [0.5008, 0.5008, 0.5598, 0.5239, 0.5164, 0.5131, 0.6069, 0.6544, 0.6554, 0.6444, 0.6740, 
                         0.6474, 0.6351, 0.6751, 0.6757, 0.6444, 0.6850, 0.6288, 0.6476, 0.6382, 0.6587, 0.6429, 
                         0.6413, 0.6710, 0.6695, 0.6413, 0.6319, 0.6351, 0.6397, 0.6898, 0.6070, 0.6102, 0.5710, 
                         0.6726, 0.6444, 0.6474, 0.6196, 0.6666, 0.6234, 0.6914, 0.7664, 0.7351, 0.7377, 0.7141, 
                         0.7679, 0.7757, 0.7757, 0.7085, 0.7476, 0.7955],   
                    "Model": ["EfficientNet-B0"] * 50 + ["ResNet50"] * 50 + ["VGG16"] * 50 + ["MobileNetV2"] * 50
                }
                epoch_df4 = pd.DataFrame(epoch_data4)
                filtered_df4 = epoch_df4[epoch_df4["Model"].isin(selected_models4)]

                fig_line4 = px.line(
                    filtered_df4,
                    x="Epoch",
                    y="Recall",
                    color="Model",
                    markers=True,
                    title="Epoch-wise Recall Progression",
                    labels={"Recall": "Recall", "Epoch": "Epoch Number"}
                )
                fig_line4.update_layout(yaxis_range=[0.5, 1.0], height=500)
                st.plotly_chart(fig_line4, use_container_width=True)



        time.sleep(6)
        with tab3:

                        
                    # Display Detection only (No image shown)
                        time.sleep(1)
                        st.success(f"Detection: {predicted_label}")
                        st.error(f"""
                    **🧪 Model Version:** `EfficientNetB0` 
                    """)
                        time.sleep(2)
                        st.markdown(predicted_sentence)
                        st.markdown("---")
                        st.toast(f" Detection: {predicted_label}")

                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        model_version = "EfficientNetB0"  # Define the model version

                        report_data = {
                            "Timestamp": [timestamp],
                            "Prediction": [predicted_label],
                            "Model Version": [model_version],
                            "Explanation": [predicted_sentence]
                        }

                        report_df = pd.DataFrame(report_data)

                        # ⬇️ Export report
                        st.subheader("⬇️ Download Detection Report")

                        csv = report_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="Download Report as CSV",
                            data=csv,
                            file_name='detection_report.csv',
                            mime='text/csv'
                        )

    else:
        st.toast("Please upload an image.",icon='📁')

def preprocess_image(image):
    # plt.imsave('image2.jpg', image)
    img_array = np.array(image)
    rgb_image = np.repeat(img_array[:, :, np.newaxis], 3, axis=2)
    img = Image.fromarray(img_array.astype('uint8'))
    # img.save('output1.jpg')  # Save the image to a file
    img_array = np.expand_dims(rgb_image, axis=0)
    return img_array

def predict(image):
    img_array = preprocess_image(image)
    prediction = model.predict(img_array)
    # print(prediction)
    predicted_idx = np.argmax(prediction, axis=1)[0]
    return predicted_idx

    

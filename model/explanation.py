"""
Legacy SHAP exploration utilities — not called by the experiment scripts.
Kept for reference only.
"""
import os
import shap
import matplotlib.pyplot as plt
import numpy as np
import torch
from model.utils import get_flight_lengths

def shap_analysis(explainableModel, X_attack_n, y_attack, flight_number, attack_index, feature_names, config):

    attack_names = ["Noise", "Landing", "Departing", "Manoeuvre", "Normal"]
    attack_name = attack_names[attack_index]
    flight_length, flight_length_sum, flight_names = get_flight_lengths(config['preprocessing']['window_size'])
    flight_name = flight_names[flight_number]

    print(f"Analysis of the flight {flight_name} under the attack {attack_name}:\n")
    
    X_explain = X_attack_n[attack_index][flight_length_sum[flight_number]:flight_length_sum[flight_number + 1], :]
    X_safe = X_attack_n[4][flight_length_sum[flight_number]:flight_length_sum[flight_number + 1], :]
    
    print(X_explain)
    #background = torch.FloatTensor(X_safe)
    background = torch.FloatTensor(np.mean(X_safe, axis=0)).unsqueeze(0)
    #background = torch.zeros(1, 50)

    y_explain = y_attack[attack_index][flight_length_sum[flight_number]:flight_length_sum[flight_number + 1]]
    
    explainer = shap.DeepExplainer(explainableModel, background)
    print("Features: ", feature_names)

    ## Keep only the anomaly, no need to explain the safe data
    indices = []
    for i in range (X_explain.shape[0]):
        y_pred = explainableModel.predict(X_explain[i,:])
        if y_pred > config['explanation']['explain_threshold']:
            indices.append(i)
    print("Anomaly detected: ", len(indices))
    X_explain = np.array([X_explain[i, :] for i in indices])
    y_explain = np.array([y_explain[i] for i in indices])

    print(torch.FloatTensor(X_explain).shape)
    shap_values = explainer(torch.FloatTensor(X_explain))
    #shap_values = explainer(background)
    
    
    shap_values.feature_names = feature_names
                                    
    print("Shap values shape: ", shap_values.shape)
    title = f"Flight : {flight_name}, Attack: {attack_name}"
    print("Flight: ", title)


    #plt.get_current_fig_manager().full_screen_toggle()
    shap.plots.initjs()
    # --------- Waterfall ----------------#

    if(config['explanation']['waterfall']):
        shap_values.base_values = torch.full((shap_values.shape[0], 1), explainer.expected_value[0])
        shap.plots.waterfall(shap_values[480,:,0], max_display = 8, show = False)
        #plt.get_current_fig_manager().full_screen_toggle()
        plt.subplots_adjust(left=0.62)
        #plt.title(title)
        folder_path = f"plots/waterfall/{config['name']}"
        plt.savefig(os.path.join(folder_path, f"{flight_name}_{attack_name}.png"), format="png", dpi=300)
        plt.show()
        plt.close()

    # --------- Beeswarm ----------------#

    if(config['explanation']['beeswarm']):
        shap.plots.beeswarm(shap_values[:,:,0], max_display=10, show = False)
        plt.subplots_adjust(left=0.44)
        plt.xlim(-1, 1)
        plt.title(title)
        folder_path = f"plots/beeswarm/{config['name']}"
        os.makedirs(folder_path, exist_ok=True)
        plt.savefig(os.path.join(folder_path, f"{flight_name}_{attack_name}.png"), format="png", dpi=300)
        plt.show()
        plt.close()

    # ----------- Bar -------------------#

    if(config['explanation']['bar']):
        shap.plots.bar(shap_values[:,:,0], show = False)
        plt.subplots_adjust(left=0.42)
        plt.xlim(0, 1)
        plt.title(title)
        folder_path = f"plots/bar/{config['name']}"
        plt.savefig(os.path.join(folder_path, f"{flight_name}_{attack_name}.png"), format="png", dpi=300)
        plt.close()

    # ----------- Scatter -------------------#


    if(config['explanation']['scatter']):
        feature_importances = np.mean(shap_values[:, :, 0].values, axis=0)
        main_feature = np.argmax(feature_importances)
        print(feature_names[main_feature])
        for i in range (len(feature_names)):
            feature = feature_names[i]
            shap.plots.scatter(shap_values[:, i, 0], color=shap_values[:,4,0], show = False)
            plt.subplots_adjust(left=0.15, right = 1)
            #plt.ylim(-1, 1)
            #plt.title(title)
            folder_path = f"plots/scatter/{config['name']}/{feature}"
            os.makedirs(folder_path, exist_ok=True)
            file_path = f"{flight_name}_{attack_name}.png"
            plt.savefig(os.path.join(folder_path, file_path), format="png", dpi=300)
            plt.show()
            plt.close()

    # ----------- Decision -------------------#

    if(config['explanation']['decision']):
        print("Expected value: ", explainer.expected_value)
        shap.decision_plot(explainer.expected_value, shap_values[:30,:,0].values, X_explain[:30,:], feature_names = np.array(feature_names), show = False)    
        plt.xlim(-1, 1)
        plt.title(title)
        plt.savefig(os.path.join(f"plots/decision/{config['name']}", f"{flight_name}_{attack_name}.png"), format="png", dpi=300)
        plt.close()


def show_plots(config):

    plot_dir = config["input_plots"]

    files = [f for f in os.listdir(plot_dir) if f.endswith('.png')]
    flight_codes = sorted(set(f.split('_')[0] for f in files))
    attack_names = sorted(set(f.split('_')[1].split('.')[0] for f in files))

    # Precise there the attacks/flight you DON'T want to plot
    excluded_flights = {"BAW9154", "BAW60T", "DLH444", "VIR63"}
    #excluded_attacks = {"Departing"}
    #excluded_flights = {"BAW9L", "CFG114", "VIR63"}
    
    flight_codes = [code for code in flight_codes if code not in excluded_flights]
    #attack_names = [code for code in attack_names if code not in excluded_attacks]
    

    fig, axs = plt.subplots(len(flight_codes), len(attack_names))

    for i, flight_code in enumerate(flight_codes):
        for j, attack_index in enumerate(attack_names):
            filename = f"{flight_code}_{attack_index}.png"
            filepath = os.path.join(plot_dir, filename)

            if os.path.exists(filepath):
                img = plt.imread(filepath)
                axs[i, j].imshow(img)
                axs[i, j].axis('off')
                #axs[i, j].set_title(f"{flight_code} - {attack_index}")

    #plt.get_current_fig_manager().full_screen_toggle()
    plt.subplots_adjust(left = 0, right = 1, bottom = 0, top = 1, wspace=0.0, hspace=0.0) 
    plt.savefig(config["output_plots"], format="png", dpi=300)
    plt.show()
    
    
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('Disease Prediction.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    global DB
    Diagnosis= {"Acne":"ISOTRETINOIN, ORAL CONTRACEPTIVES, TOPICAL ANTIMICROBIALS, ORAL ANTIBIOTICS, CORTICOSTEROID INJECTION, TOPICAL ANTIBIOTICS, COUNTER MEDICINES","Fungal infection":"CORTICOSTEROIDS, TOPICAL DRUGS, ANTIFUNGAL DRUGS, POWDERS","Allergy":"EMERGENCY EPINEPHRINE, IMMUNOTHERAPY, NASAL SPRAY","GERD":"PROTON PUMP INHIBITORS, H2 BLOCKERS, ANTACIDS","Chronic cholestasis":"URSODIOL, HOME TREATMENT","Drug Reaction":"INJECTIONS, ANTIHISTAMINE, DISCONTINUE DRUG","Peptic ulcer diseae":"PROTON PUMP INHIBITORE, HISTAMINE(H2) BLOCKERS, ANTACIDS","AIDS":"NO TREATMENT","Diabetes ":"SURGERY, INSULIN INJECTIONS, ORAL MEDICATIONS","Gastroenteritis":"ANTIBIOTICS, FLUID REPLACEMENT , ORAL REHYDRATION DRINKS","Bronchial Asthma":"LEUKOTRIENE MODIFIERS, INHALED CORTICOSTEROIDS, BRONCHODILATORS","Hypertension ":"CALCIUM CHANNEL BLOCKERS, ACE INHIBITORS, DIURETICS","Migraine":"BOTOX INJECTIONS, ANTIDEPRESSANTS, BP LOWERING MEDICATIONS","Cervical spondylosis":"ANTI-SEIZURE MEDICATIONS, MUSCLE RELAXANTS, NSAIDS","Paralysis (brain hemorrhage)":"SURGERY, ANTI-EPILEPTIC DRUGS, REHABILITATION","Jaundice":"INTRAVENOUS, PHOTOTHERAPY, REHYDRATION FLUIDS","Malaria":"ANTIMALARIAL DRUGS","Chicken pox":"ANTIVIRAL DRUGS, SKIN LOTIONS, HOME TREATMENT","Dengue":"EMERGENCY ADMISSION, ORAL DRUGS, HOME TREATMENT","Typhoid":"MEDICATIONS","hepatitis A":"NO TREATMENT","Hepatitis B":"NO TREATMENT","Hepatitis C":"ANTIVIRAL MEDICATIONS","Hepatitis D":"PEGYLATED INTERFERON ALPHA","Hepatitis E":"NO TREATMENT","Alcoholic hepatitis":"GLUCOCORTICOIDS, CORTICOSTEROIDS, QUITTING DRINKING","Tuberculosis":"6-MONTH RIPE TREATMENT, 4-MONTH RIFAPENTINE-MOXIFLOXACIN REGIMEN","Common Cold":"DECONGESTANT NASAL SPRAYS, PAIN KILLERS, HOME REMEDIES","Pneumonia":"OXYGEN THERAPY, NSAIDS, HOME REMEDIES","Dimorphic hemmorhoids(piles)":"NON EXCISIONAL OPERATION, TOPICAL OR SYSTEMIC MEDICATION, DIETARY AND LIFESTYLE MODIFICATION","Heart attack":"SURGICAL PROCEDURES, BETA BLOCKERS, BLOOD THINNING MEDICATIONS","Varicose veins":"AMBULATORY PHLEBECTOMY, LASER TREATMENT, SCLEROTHERAPY","Hypothyroidism":"THYROID HORMONE THERAPY, THYROID HORMONE THERAPY, SYNTHETIC THYROID HORMONE","Hyperthyroidism":"SURGERY, ANTI-THYROID MEDICATIONS, RADIOACTIVE IODINE","Hypoglycemia":"INTRAVENOUS GLUCOSE, GLUCAGON INJECTION, EAT/DRINK FAST ACTING CARBOHYDRATES","Osteoarthristis":"DULOXETINE, NSAIDS, ACETAMINOPHEN","Arthritis":"DMARDS, CORTICOSTEROID MEDICATIONS, NSAIDS","(vertigo) Paroymsal  Positional Vertigo":"SURGERY, ANTI-ANXIETY MEDICATIONS, NASEAU RELIEF MEDICATIONS","Urinary tract infection":"INTRAVENOUS ANTIBIOTICS, LOW-DOSE ANTIBIOTICS, TRIMETHOPRIM/SULFAMETHOXAZOLE","Psoriasis":"CALCINEURIN INHIBITORS, VITAMIN D ANALOGUES, CORTICOSTEROIDS","Impetigo":"ORAL ANTIBIOTICS, ANTIBIOTIC OINTMENTS"}
    Symp1= ["itching","skin_rash","nodal_skin_eruptions","continuous_sneezing","shivering","chills","joint_pain","stomach_pain","acidity","ulcers_on_tongue","muscle_wasting","vomiting","burning_micturition","spotting_ urination","fatigue","weight_gain","anxiety","mood_swings","weight_loss","restlessness","lethargy","irregular_sugar_level","cough","high_fever","sunken_eyes","breathlessness","sweating","indigestion","headache","yellowish_skin","dark_urine","nausea","loss_of_appetite","pain_behind_the_eyes","back_pain","constipation","abdominal_pain","diarrhoea","mild_fever","yellow_urine","yellowing_of_eyes","acute_liver_failure","fluid_overload","swelling_of_stomach","swelled_lymph_nodes","malaise","blurred_and_distorted_vision","phlegm","throat_irritation","chest_pain","weakness_in_limbs","fast_heart_rate","pain_during_bowel_movements","neck_pain","dizziness","cramps","obesity","knee_pain","muscle_weakness","stiff_neck","swelling_joints","movement_stiffness","spinning_movements","loss_of_balance","weakness_of_one_body_side","bladder_discomfort","passage_of_gases","toxic_look_(typhos)","depression","irritability","muscle_pain","red_spots_over_body","family_history","mucoid_sputum","rusty_sputum","lack_of_concentration","visual_disturbances","blood_in_sputum","pus_filled_pimples","skin_peeling","blister"]
    int_features = request.form.getlist("Symptoms")
    print(int_features)
    l2=[]
    for i in Symp1:
        if i in int_features:
            l2.append(1)
        else:
            l2.append(0)
        
    symp_dict = {Symp1[i]: l2[i] for i in range(len(Symp1))}

    DB=pd.DataFrame(symp_dict,index=[0]) 
    prediction = model.predict(DB)

    output = " ".join(prediction)
    for k,v in Diagnosis.items():
        if k==output:
            treatment= v



    return render_template('index.html', prediction_text='The predicted disease is {}. \n'.format(output),treatment_text="\n The treatments for the {} are: {}.".format(output,treatment))

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict(pd.DataFrame(data,index=[0]))

    output = " ".join(prediction)
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)
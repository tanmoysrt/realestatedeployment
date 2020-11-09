from django.http.response import HttpResponse
from django.shortcuts import render
import pickle
import numpy as np
import pandas as pd
from realestatedeployment.settings import BASE_DIR
import os
model_path = os.path.join(BASE_DIR,os.path.join('mainapp','banglore_home_prices_model.pickle'))
columns_list =['total_sqft', 'bath', 'bhk','1st Block Jayanagar', '1st Phase JP Nagar',
       '2nd Phase Judicial Layout', '2nd Stage Nagarbhavi',
       '5th Block Hbr Layout', '5th Phase JP Nagar', '6th Phase JP Nagar',
       '7th Phase JP Nagar', '8th Phase JP Nagar', '9th Phase JP Nagar',
       'AECS Layout', 'Abbigere', 'Akshaya Nagar', 'Ambalipura',
       'Ambedkar Nagar', 'Amruthahalli', 'Anandapura', 'Ananth Nagar',
       'Anekal', 'Anjanapura', 'Ardendale', 'Arekere', 'Attibele',
       'BEML Layout', 'BTM 2nd Stage', 'BTM Layout', 'Babusapalaya',
       'Badavala Nagar', 'Balagere', 'Banashankari',
       'Banashankari Stage II', 'Banashankari Stage III',
       'Banashankari Stage V', 'Banashankari Stage VI', 'Banaswadi',
       'Banjara Layout', 'Bannerghatta', 'Bannerghatta Road',
       'Basavangudi', 'Basaveshwara Nagar', 'Battarahalli', 'Begur',
       'Begur Road', 'Bellandur', 'Benson Town', 'Bharathi Nagar',
       'Bhoganhalli', 'Billekahalli', 'Binny Pete', 'Bisuvanahalli',
       'Bommanahalli', 'Bommasandra', 'Bommasandra Industrial Area',
       'Bommenahalli', 'Brookefield', 'Budigere', 'CV Raman Nagar',
       'Chamrajpet', 'Chandapura', 'Channasandra', 'Chikka Tirupathi',
       'Chikkabanavar', 'Chikkalasandra', 'Choodasandra', 'Cooke Town',
       'Cox Town', 'Cunningham Road', 'Dasanapura', 'Dasarahalli',
       'Devanahalli', 'Devarachikkanahalli', 'Dodda Nekkundi',
       'Doddaballapur', 'Doddakallasandra', 'Doddathoguru', 'Domlur',
       'Dommasandra', 'EPIP Zone', 'Electronic City',
       'Electronic City Phase II', 'Electronics City Phase 1',
       'Frazer Town', 'GM Palaya', 'Garudachar Palya', 'Giri Nagar',
       'Gollarapalya Hosahalli', 'Gottigere', 'Green Glen Layout',
       'Gubbalala', 'Gunjur', 'HAL 2nd Stage', 'HBR Layout',
       'HRBR Layout', 'HSR Layout', 'Haralur Road', 'Harlur', 'Hebbal',
       'Hebbal Kempapura', 'Hegde Nagar', 'Hennur', 'Hennur Road',
       'Hoodi', 'Horamavu Agara', 'Horamavu Banaswadi', 'Hormavu',
       'Hosa Road', 'Hosakerehalli', 'Hoskote', 'Hosur Road', 'Hulimavu',
       'ISRO Layout', 'ITPL', 'Iblur Village', 'Indira Nagar', 'JP Nagar',
       'Jakkur', 'Jalahalli', 'Jalahalli East', 'Jigani',
       'Judicial Layout', 'KR Puram', 'Kadubeesanahalli', 'Kadugodi',
       'Kaggadasapura', 'Kaggalipura', 'Kaikondrahalli',
       'Kalena Agrahara', 'Kalyan nagar', 'Kambipura', 'Kammanahalli',
       'Kammasandra', 'Kanakapura', 'Kanakpura Road', 'Kannamangala',
       'Karuna Nagar', 'Kasavanhalli', 'Kasturi Nagar', 'Kathriguppe',
       'Kaval Byrasandra', 'Kenchenahalli', 'Kengeri',
       'Kengeri Satellite Town', 'Kereguddadahalli', 'Kodichikkanahalli',
       'Kodigehaali', 'Kodigehalli', 'Kodihalli', 'Kogilu', 'Konanakunte',
       'Koramangala', 'Kothannur', 'Kothanur', 'Kudlu', 'Kudlu Gate',
       'Kumaraswami Layout', 'Kundalahalli', 'LB Shastri Nagar',
       'Laggere', 'Lakshminarayana Pura', 'Lingadheeranahalli',
       'Magadi Road', 'Mahadevpura', 'Mahalakshmi Layout', 'Mallasandra',
       'Malleshpalya', 'Malleshwaram', 'Marathahalli', 'Margondanahalli',
       'Marsur', 'Mico Layout', 'Munnekollal', 'Murugeshpalya',
       'Mysore Road', 'NGR Layout', 'NRI Layout', 'Nagarbhavi',
       'Nagasandra', 'Nagavara', 'Nagavarapalya', 'Narayanapura',
       'Neeladri Nagar', 'Nehru Nagar', 'OMBR Layout', 'Old Airport Road',
       'Old Madras Road', 'Padmanabhanagar', 'Pai Layout', 'Panathur',
       'Parappana Agrahara', 'Pattandur Agrahara', 'Poorna Pragna Layout',
       'Prithvi Layout', 'R.T. Nagar', 'Rachenahalli',
       'Raja Rajeshwari Nagar', 'Rajaji Nagar', 'Rajiv Nagar',
       'Ramagondanahalli', 'Ramamurthy Nagar', 'Rayasandra',
       'Sahakara Nagar', 'Sanjay nagar', 'Sarakki Nagar', 'Sarjapur',
       'Sarjapur  Road', 'Sarjapura - Attibele Road',
       'Sector 2 HSR Layout', 'Sector 7 HSR Layout', 'Seegehalli',
       'Shampura', 'Shivaji Nagar', 'Singasandra', 'Somasundara Palya',
       'Sompura', 'Sonnenahalli', 'Subramanyapura', 'Sultan Palaya',
       'TC Palaya', 'Talaghattapura', 'Thanisandra', 'Thigalarapalya',
       'Thubarahalli', 'Thyagaraja Nagar', 'Tindlu', 'Tumkur Road',
       'Ulsoor', 'Uttarahalli', 'Varthur', 'Varthur Road', 'Vasanthapura',
       'Vidyaranyapura', 'Vijayanagar', 'Vishveshwarya Layout',
       'Vishwapriya Layout', 'Vittasandra', 'Whitefield',
       'Yelachenahalli', 'Yelahanka', 'Yelahanka New Town', 'Yelenahalli',
       'Yeshwanthpur']


def landingpage(request):
    data={
        "columns_list":columns_list[3:]
    }
    if request.method == "POST":
        sqft = request.POST.get("sqft")
        bhk = request.POST.get("bhk")
        bathroom = request.POST.get("bathroom")
        location = request.POST.get("location")
        data['sqft']=sqft
        data['bhk']=bhk
        data['bathroom']=bathroom
        data['location']=location
        data['predicted'] = round(predict_price(location,sqft, bathroom, bhk),2)

    return render(request,"mainapp/landingpage.html",data)

model = pickle.load(open(model_path,'rb'))

def predict_price(location,sqft,bath,bhk):    
    columns=pd.DataFrame(columns_list)
    # print(len(columns))
    x = np.zeros(len(columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if location in columns_list:
        loc_index = np.where(columns==location)[0][0]
        # print(loc_index)
        if loc_index >= 0:
            x[loc_index] = 1
    return model.predict([x])[0]
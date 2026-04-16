# Ketik "python" di terminal, lalu paste kode ini:
import zipfile, json

# Ganti dengan nama file ZIP mu yang dapat 0.00
zip_name = "/home/tilakoid/selectedtopics/visual_recognition-hw2/submission/submission_q1000_bs24_lr1e-4_wd1e-3_best.zip" 

with zipfile.ZipFile(zip_name, 'r') as z:
    file_list = z.namelist()
    print("Isi file dalam ZIP:", file_list)
    
    # Baca file JSON pertama di dalamnya
    if file_list:
        with z.open(file_list[0]) as f:
            data = json.load(f)
            print("Total Prediksi:", len(data))
            if len(data) > 0:
                print("Contoh 1 Prediksi:", data[0])
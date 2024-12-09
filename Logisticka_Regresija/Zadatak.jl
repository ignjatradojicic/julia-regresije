# LOGISTIČKA REGRESIJA
# FAJL: user_behavior.csv

# Predvideti da li je operativni sistem Android ili ne.
# Podatke podeliti u odnosu 80:20.
# Ispitati osetljivost, preciznost i specificnost, kao i ROC i AUC.

using Statistics
using StatsModels
using StatsBase
using GLM
using DataFrames
using CSV
using Plots
using StatsPlots
using ROC
using MLBase
using Lathe

# Ucitavanje datoteke
data = DataFrame(CSV.File("user_behavior.csv"))

# Prikaz ucitanih podataka
display(describe(data))

# Analiza podataka
# Kolona User Behavior Class preko 60% nedostajucih podataka pa je izbacujemo
select!(data, Not([:User_Behavior_Class]))

# Prikaz izmenjenih podataka
display(describe(data))

# Ako je broj podataka koji nedostaju sa malim udelom, onda te podatke možemo
# zameniti stanjem koje se najčešće pojavljuje u data set-u
data[ismissing.(data[!, :Operating_System]), :Operating_System] .= mode(skipmissing(data[!, :Operating_System]))

# Ako nedostaje malo veci broj podataka numerickog tipa, popunicemo prosecnom vrednoscu
data[ismissing.(data[!, :"App_Usage_Time"]), :"App_Usage_Time"] .= trunc(Int64, mean(skipmissing(data[!, :"App_Usage_Time"])))

# Prikaz izmenjenih podataka
display(describe(data))

# Kolone imaju veci broj nedostajucih redova (5% nedostajucih redova), pa cemo te redove izbaciti
dropmissing!(data, [:Screen_On_Time, :Battery_Drain, :Number_of_Apps_Installed])

# Prikaz izmenjenih podataka
display(describe(data))

# Grafik zavisne i nezavisne promenljive
scatter(data.Screen_On_Time, data.Operating_System)

# Uklonicemo sve prevelike vrednosti koje kvare analizu
# Broj provedenih sati na telefonu ne moze biti veci od duzine dana
filter!(row -> row.Screen_On_Time <= 24, data)

# Grafik zavisne i nezavisne promenljive - provera
scatter(data.Screen_On_Time, data.Operating_System)

# Prikaz izmenjenih podataka
display(describe(data))

# Prikaz broja instaliranih aplikacija
scatter(data.Number_of_Apps_Installed, data.Operating_System)

# Broj instaliranih aplikacija u proseku nije veci od 100
filter!(row -> row.Number_of_Apps_Installed <= 100, data)

# Provera putem grafika
scatter(data.Number_of_Apps_Installed, data.App_Usage_Time)

# Prikaz izmenjenih podataka
display(describe(data))

# Modelovanje veze izmedju zavisnih i nezavisnih promenljivih
fm = @formula(Operating_System ~ Device_Model + App_Usage_Time + Screen_On_Time + Battery_Drain + Number_of_Apps_Installed + Data_Usage + Age + Gender)

# Podela podataka na skup za obuku i skup za validaciju
dataTrain, dataTest = Lathe.preprocess.TrainTestSplit(data, 0.80)

# Kreiranje regresionog modela
model = glm(fm, dataTrain, Binomial(), LogitLink())

# Predvidjanje vrednosti
predictedTest = predict(model, dataTest)

# Kreiranje prazne kolekcije za fit vrednosti predikecije na skup 0 ili 1
predictedClass = repeat(0:0, length(predictedTest))

for i in 1:length(predictedTest)
    if predictedTest[i] < 0.5
        predictedClass[i] = 0
    else
        predictedClass[i] = 1
    end
end

# FP, FN, TP, TN
FPTest = 0
FNTest = 0
TPTest = 0
TNTest = 0

for i in 1:length(predictedClass)
    if dataTest.Operating_System[i] == 1 && predictedClass[i] == 1
        global TPTest+=1
    elseif dataTest.Operating_System[i] == 0 && predictedClass[i] == 0
        global TNTest +=1
    elseif dataTest.Operating_System[i] == 0 && predictedClass[i] == 1
        global FPTest +=1
    elseif dataTest.Operating_System[i] == 1 && predictedClass[i] == 0
        global FNTest +=1
    end    
end

# Racunanje preciznosti, osetljivosti i specificnosti
accuracy = (TPTest+TNTest)/(TPTest+TNTest+FPTest+FNTest)
sensitivity = (TPTest)/(TPTest+FNTest)
sensitivity = (TNTest)/(TNTest+FPTest)

println("\naccuracy : $accuracy")
println("sensitivity : $sensitivity")
println("sensitivity : $sensitivity")

# Povrsina ispod ROC krive
rocTest = ROC.roc(predictedTest, dataTest.Operating_System, true)
aucTest = AUC(rocTest)

# Ocena kvaliteta klasifikatora
if aucTest > 0.9
    println("Klasifikator je jako dobar")
elseif aucTest > 0.8
    println("Klasifikator je veoma dobar")
elseif aucTest > 0.7
    println("Klasifikator je dosta dobar")
elseif aucTest > 0.5
    println("Klasifikator je relativno dobar")
else
    println("Klasifikator je los")
end
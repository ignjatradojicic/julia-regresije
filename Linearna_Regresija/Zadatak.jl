# LINEARNA REGRESIJA
# FAJL: bowling_summaries.csv

# Predvideti economy za svaki kuglacki tim.
# Podatke podeliti u odnosu 80:20.
# Izracunati prosecnu relativnu, prosecnu apsolutnu, MSE i RMSE gresku.
# Oceniti kvalitet modela pomocu r2.
# Proveriti koeficijent korelacije izmedju nezavisne promenljive i zavisne.

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
data = DataFrame(CSV.File("bowling_summaries.csv"))

# Prikaz ucitanih podataka
display(describe(data))

# Analiza podataka
# Kolona Match_Id preko 70% nedostajucih podataka pa je izbacujemo
select!(data, Not([:Match_Id]))

# Prikaz izmenjenih podataka
display(describe(data))

# Ako je broj podataka koji nedostaju sa malim udelom, onda te podatke možemo
# zameniti stanjem koje se najčešće pojavljuje u data set-u
data[ismissing.(data[!, :match]), :match] .= mode(skipmissing(data[!, :match]))
data[ismissing.(data[!, :wides]), :wides] .= mode(skipmissing(data[!, :wides]))
data[ismissing.(data[!, :noBalls]), :noBalls] .= mode(skipmissing(data[!, :noBalls]))

# Prikaz izmenjenih podataka
display(describe(data))

# Ako nedostaje malo veci broj podataka numerickog tipa, popunicemo prosecnom vrednoscu
data[ismissing.(data[!, :"s6_Conceded"]), :"s6_Conceded"] .= trunc(Int64, mean(skipmissing(data[!, :"s6_Conceded"])))

# Prikaz izmenjenih podataka
display(describe(data))

# Sve redove za koje kolone imaju preko 40 nedostajucih vrednosti izbacujemo
dropmissing!(data, [:runsConceded, :wickets, :economy])

# Prikaz izmenjenih podataka
display(describe(data))

# koeficijent korelacije
if cor(data.economy, data.runsConceded) > 0.5
    println("Postoji jaka veza")
end

# Kolone runsConceded, dotBalls i s4_Conceded imaju nerealne vrednosti
scatter(data.economy, data.runsConceded)
scatter(data.economy, data.dotBalls)
scatter(data.economy, data."s4_Conceded")

# Potrebno je te vrednosti izbaciti iz analize
filter!(row -> row.runsConceded <= 500 && row.dotBalls <= 50 && row."s4_Conceded" <= 10, data)

# Prikaz izmenjenih podataka
display(describe(data))

# Modelovanje veze izmedju zavisnih i nezavisnih promenljivih
fm = @formula(economy ~ match + bowlingTeam + overs + maiden + runsConceded + wickets + dotBalls + wides + noBalls)

# Podela podataka na skup za obuku i skup za validaciju
dataTrain, dataTest = Lathe.preprocess.TrainTestSplit(data, 0.70)

# Kreiranje linarnog modela
model = lm(fm, dataTrain)

# Predvidjanje vrednosti
predictedTest = predict(model, dataTest)

# Racunanje greske
errorsTest = dataTest.economy .- predictedTest

# Prosecna relativna greska
avg_rel_error = mean(abs.(errorsTest / dataTest.economy))
println("Prosecna relativna greska = $avg_rel_error")

# Prosecna apsolutna greska
avg_abs_error = mean(abs.(errorsTest))
println("Prosecna apsolutna greska = $avg_abs_error")

errorsTestSquared = errorsTest .^ 2

# MSE
mse = mean(errorsTestSquared)
println("RMSE = $mse")

# RMSE
rmse = sqrt(mse)
println("RMSE = $rmse")

# rsquared
rsquared = r2(model)
println("\nr2 = $rsquared")

if rsquared > 0.5
    println("Model je dovoljno dobar za predvidjanje")
else
    println("Model nije dobar za predvidjanje")
end
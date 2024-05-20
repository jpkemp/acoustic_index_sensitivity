save_tropical_sound <- function(filename){
    library(soundecology)
    data(tropicalsound)
    tuneR::writeWave(tropicalsound, filename, FALSE)
}
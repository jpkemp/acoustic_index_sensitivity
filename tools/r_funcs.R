save_tropical_sound <- function(filename){
    library(soundecology)
    data(tropicalsound)
    tuneR::writeWave(tropicalsound, filename, FALSE)
}

get_aci_model_priors <- function(index) {
    priors <- c(brms::prior(normal(0, 2), class='Intercept'),
                brms::prior(normal(1e20, 100), class='b'),
                brms::prior(gamma(100, 0.01), class='shape')
    )

}

get_family <- function(index) {
    if (index == "ACI"){return(stats::Gamma(link=log))}
    if (index == "ADI"){return(stats::Gamma(link=log))}
    if (index == "AEI"){return(brms::Beta())}
    # if (index == "BIO"){return(brms::hurdle_gamma(link=log))}
    if (index == "BIO"){return(stats::Gamma(link=log))}
    # if (index == "BIO"){return(brms::hurdle_lognormal())}
}

terra_hurdle_formula = brms::bf(Value ~ (Site * Window) + (1 | Day), hu ~ (Site * Window))
# terra_hurdle_formula = brms::bf(Value ~ (Site * Window), hu ~ (Site * Window))
terra_general_formula = "Value ~ (Site * Window) + (1 | Day)"
marine_hurdle_formula = brms::bf(Value ~ (Hour * Window) + (1 | Site), hu ~ (Hour * Window) + (1 | Site))
marine_general_formula = "Value ~ (Hour * Window) + (1 | Site)"
generate_model <- function(data, family, iter, warmup, marine=TRUE, link_hu=FALSE) {
    if (link_hu) {
        if (marine) {
            formula = marine_hurdle_formula
        } else {
            formula = terra_hurdle_formula
        }
    } else {
        if (marine) {
            formula = marine_general_formula
        } else {
            formula = terra_general_formula
        }
    }

    model <- brms::brm(formula,
              data=data,
              family=family,
              iter=iter,
              warmup=warmup,
              cores=5,
              chains=5,
              init=0,
              open_progress=FALSE,
              silent=TRUE
            )
}
find_effects <- function(data, index, output_path, marine, iter=5000, warmup=4000) {
    data_path <- paste(output_path, "_data.RData", sep="")
    save(data, file=data_path)
    family <- get_family(index)
    # model <- generate_model(data, family, iter=iter, warmup=warmup, marine=marine, link_hu=index=="BIO")
    model <- generate_model(data, family, iter=iter, warmup=warmup, marine=marine, link_hu=FALSE)
    model_path <- paste(output_path, "_model.RData", sep="")
    save(model, file=model_path)
    effects <- brms::conditional_effects(model)
    effects_path <- paste(output_path, "_effects.RData", sep="")
    save(effects, file=effects_path)

    return(list(model, effects))
}

save_rdf <- function(data, file_path) {
    save(data, file=file_path)
}

get_posterior_ratios <- function(model, num="12", den="0", cross="Hour") {
    nd <- model$data |>
    dplyr::distinct(Window, !!dplyr::sym(cross))
    posterior_ratios <- brms::posterior_epred(
    model, newdata = nd, re_formula = NA
    ) |>
    t() |>
    data.frame(check.names = FALSE) |>
    dplyr::bind_cols(nd) |>
    tidyr::pivot_longer(cols = tidyselect::matches("\\d+"), values_to = "value", names_to = ".draw") |>
    # tidyr::pivot_longer(cols = 1:4000, values_to = "value", names_to = ".draw") |>
    dplyr::mutate(.draw = as.numeric(.draw)) |>
    dplyr::group_by(Window, .draw) |>
    dplyr::summarise(ratio = value[!!dplyr::sym(cross) == num] / value[!!dplyr::sym(cross) == den]) |> 
    dplyr::ungroup()
}

# calc_dist_same_ratio <- function(x, thresh=0.1) {
#  sum(x > -thresh & x < thresh) / length(x)
# }

get_posterior_ratio_difference <- function(posterior_ratios, num="12", den="0", cross="Hour") {
    posterior_ratios |>
    dplyr::group_by((.draw)) |>
    dplyr::summarise(difference = max(ratio) / min(ratio)) |>
    # dplyr::summarise(exc_prob = calc_dist_same_ratio(difference)) |>
    dplyr::summarise(mean = mean(difference), sd=sd(difference)) |>
    dplyr::ungroup()
}

get_posterior_ratio_summary <- function(posterior_ratios) {
    posterior_ratios |>
    dplyr::group_by(Window) |> # group by draw instead
    # calculate difference of ratio between two windows -> as per 91, but Window instead; probability of > or < 0 should be similar
    dplyr::summarise(ggdist::median_hdci(ratio)) |>
    dplyr::ungroup()
}

get_warnings <- function() {
    return(warnings())
}
# Assume working directory is GPD_ScaleMixture/

# Load data
load("./data/realdata/GHCNdailyPrecip_centralUS_1949_2023.RData")
stations_full <- stations
rm(stations)

# Subset to JJA observations
# JJA_indices <- which(time_info$raw.month %in% c(6, 7, 8))
# Y_daily_JJA <- t(precip_mat[JJA_indices, ]) # we want to have a matrix of size (Ns, Nt)
# dim(Y_daily_JJA)

# Select 90 summer days from 6-22 to 9-19

# Convert time_info$time to Date object if not already
time_info$time <- as.Date(time_info$time, format = "%Y-%m-%d")
# Extract month and day information from the Date object
time_info$month_day <- format(time_info$time, "%m-%d")
# Get the indices of the rows where the date is between June 22 and September 19
summer_indices <- which(time_info$month_day > "06-21" & time_info$month_day < "09-20")

Y_daily_summer <- t(precip_mat[summer_indices, ])
dim(Y_daily_summer)



# Truncate the data:
#   1. Break the matrix Y_daily_JJA into 75 chunks (75 years), 
#     each representing a 3-month period of summer from 6-22 to 9-19 (to perfectly match 90 days).
#   2. For each chunk (which has 90 days), break it into 9 pieces of 10-day blocks
#   3. Take the maximum of each of these 9 pieces.
#   4. Combine the results.

# Number of days in each year (June, July, August)
# JJA_days <- 92
summer_days <- 90
# Number of years
n_years <- 75


# Create a function to split each 3-month chunk into 9 pieces and take the maximum of each piece
process_monthly_chunk <- function(month_data) {
    # Number of days in 3 months (June, July, August)
    n_days <- ncol(month_data)
    # Calculate the number of days in each of the 9 pieces
    piece_size <- floor(n_days / 9)

    # Split the 3-month data into 9 pieces and take the maximum of each piece
    max_values <- sapply(1:9, function(i) {
        # Define the start and end index for each piece
        start_idx <- (i - 1) * piece_size + 1
        end_idx <- i * piece_size
        # if (i == 9) {
        #     # The last piece should include all remaining days
        #     end_idx <- n_days
        # } else {
        #     end_idx <- i * piece_size
        # }
        if(end_idx - start_idx + 1 != 10){
            print("size not 10!")
        }
        # Take the maximum of this piece across each location
        apply(month_data[, start_idx:end_idx], 1, max)
    })

    # Return the result as a matrix with 9 columns (one for each piece)
    return(max_values)
}

# Initialize a list to store results
season_maxs <- list()

# Loop over each year and process the data
for (i in 1:n_years) {
    # Extract the data for each 3-month chunk (92 days)
    start_day <- (i - 1) * summer_days + 1
    end_day <- i * summer_days
    year_data <- Y_daily_summer[, start_day:end_day]

    # Process the monthly chunk and get the max values for each of the 9 pieces
    season_maxs[[i]] <- process_monthly_chunk(year_data)
}

# Combine the results from all years
Y <- do.call(cbind, season_maxs)
dim(Y)


# HOW TO GET GP ESTIMATES?
#   for each single station, fit a GP

library(extRemes)
extract_gpd_parameters <- function(data, threshold_prob = 0.95) {
    # Initialize lists to store shape and scale parameters
    threshold_params <- numeric(nrow(data))
    shape_params <- numeric(nrow(data))
    scale_params <- numeric(nrow(data))

    for (i in 1:nrow(data)) {
        # Get data for each site
        site_data <- na.omit(data[i, ]) # omit missing values in data[i, ]

        # Define the threshold for top 5% of observations
        threshold <- quantile(site_data, threshold_prob)

        # Extract the data above the threshold
        exceedances <- site_data[site_data > threshold]

        threshold_params[i] <- threshold
        # Fit the GPD to the exceedances
        if (length(exceedances) > 0) {
            #   fit <- gpd.fit(exceedances - threshold)
            fit <- fevd(exceedances - threshold, type = "GP", threshold = 0)

            # Extract shape and scale parameters
            shape_params[i] <- fit$results$par[2] # Shape parameter (xi)
            scale_params[i] <- fit$results$par[1] # Scale parameter (sigma)
        } else {
            shape_params[i] <- NA
            scale_params[i] <- NA
        }
    }

    # Return the parameters as a list
    return(list(thresholds = threshold_params, shape = shape_params, scale = scale_params))
}

gpd_results <- extract_gpd_parameters(Y)


# Save into one data object

GP_estimates <- data.frame(
    "mu" = gpd_results$thresholds,
    "logsigma" = log(gpd_results$scale),
    "xi" = gpd_results$shape
)

elev <- stations_full$elevation
stations <- data.frame("x" = stations_full$longitude, "y" = stations_full$latitude)

save(Y, GP_estimates, elev, stations, file = "./data/realdata/JJA_precip_nonimputed.RData")

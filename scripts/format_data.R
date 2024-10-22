# Assume working directory is GPD_ScaleMixture/

# Load data
load('./data/realdata/GHCNdailyPrecip_centralUS_1949_2023.RData')
stations_full <- stations
rm(stations)

# Subset to JJA observations

JJA_indices <- which(time_info$raw.month %in% c(6,7,8))

Y_daily_JJA <- t(precip_mat[JJA_indices,]) # we want to have a matrix of size (Ns, Nt)
dim(Y_daily_JJA)

# Truncate the data:
#   1. Break the matrix Y_daily_JJA into 75 chunks, each representing a 3-month period (June, July, August).
#   2. For each chunk (which has 92 days), break it into 9 pieces.
#   3. Take the maximum of each of these 9 pieces.
#   4. Combine the results.

# Number of days in each year (June, July, August)
JJA_days <- 92
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
    if (i == 9) {
      # The last piece should include all remaining days
      end_idx <- n_days
    } else {
      end_idx <- i * piece_size
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
  start_day <- (i - 1) * JJA_days + 1
  end_day <- i * JJA_days
  year_data <- Y_daily_JJA[, start_day:end_day]
  
  # Process the monthly chunk and get the max values for each of the 9 pieces
  season_maxs[[i]] <- process_monthly_chunk(year_data)
}

# Combine the results from all years
Y_JJA <- do.call(cbind, season_maxs)
dim(Y_JJA)


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

# Run the function on your data
gpd_results <- extract_gpd_parameters(Y_JJA)

# Save into data object

elev <- stations$elevation
stations <- data.frame('x' = stations_full$longtitude, 'y' = stations_full$latitude)





GP_fit <- fevd(na.omit(Y_JJA[1,]), threshold = quantile(na.omit(Y_JJA[1,]), 0.95, na.rm=TRUE), type="GP")
GP_fit2 <- fevd(na.omit(Y_JJA[1,]) - quantile(na.omit(Y_JJA[1,]), 0.95), type="GP", threshold = 0)

#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import the required libraries
import gatspy
import scipy
import math
import glob

import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)

from os import path
from gatspy import datasets, periodic

import warnings
warnings.filterwarnings("ignore")

from matplotlib import pyplot as plt, ticker as mticker, colors
from matplotlib.ticker import FormatStrFormatter

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel as W

from astropy.stats import sigma_clip
from astropy import units as u 
from astropy.coordinates import SkyCoord, EarthLocation
from astropy import log
from astropy.table import Table
from astropy import time, coordinates as coord, units as u
from astroquery.xmatch import XMatch


# In[2]:


# Function that converts flux (Jy) to magnitude
def flux2mag(flux, dflux = [], flux_0 = 3631.0):

    # Create the mask for flux <= 0
    mask_flux = (flux <= 0)
    
    # Remove the flux values that are non-positive
    flux = flux[mask_flux == False]
    
    mag = -2.5 * np.log10(flux / flux_0)

    if dflux == []:
        return mag

    else:
        dmag_p = -2.5 * np.log10((flux - dflux) / flux_0) - mag
        dmag_n = -2.5 * np.log10((flux + dflux) / flux_0) - mag

    return mag, dmag_p, dmag_n


# In[3]:


# Function for calculating the absolute magnitude
def absolute_magnitude(apparent_magnitude, parallax, parallax_error):
    
    abs_mag  = apparent_magnitude + 5 * (np.log10(parallax * 0.001)+ 1)
    
    abs_mag_p  = apparent_magnitude + 5 * (np.log10((parallax + parallax_error) * 0.001) + 1)

    mag_error = abs_mag_p - abs_mag
    
    return abs_mag, mag_error


# In[4]:


# Function for getting Gaia data
def send_radial_gaia_query(query_size=1000000, distance=200, **kwargs):
    """
    Sends an archive query for d < 200 pc, with additional filters taken from
    Gaia Data Release 2: Observational Hertzsprung-Russell diagrams (Sect. 2.1)
    Gaia Collaboration, Babusiaux et al. (2018)
    (https://doi.org/10.1051/0004-6361/201832843)

    NOTE: 10000000 is a maximum query size (~76 MB / column)

    Additional keyword arguments are passed to TapPlus.launch_job_async method.
    """
    from astroquery.utils.tap.core import TapPlus

    gaia = TapPlus(url="https://gea.esac.esa.int/tap-server/tap")

    job = gaia.launch_job_async("select top {}".format(query_size)+
                #" lum_val, teff_val,"
                #" ra, dec, parallax,"
                " bp_rp, phot_g_mean_mag+5*log10(parallax)-10 as mg"
         " from gaiadr2.gaia_source"
         " where parallax_over_error > 10"
         " and visibility_periods_used > 8"
         " and phot_g_mean_flux_over_error > 50"
         " and phot_bp_mean_flux_over_error > 20"
         " and phot_rp_mean_flux_over_error > 20"
         " and phot_bp_rp_excess_factor <"
            " 1.3+0.06*power(phot_bp_mean_mag-phot_rp_mean_mag,2)"
         " and phot_bp_rp_excess_factor >"
            " 1.0+0.015*power(phot_bp_mean_mag-phot_rp_mean_mag,2)"
         " and astrometric_chi2_al/(astrometric_n_good_obs_al-5)<"
            "1.44*greatest(1,exp(-0.4*(phot_g_mean_mag-19.5)))"
         +" and 1000/parallax <= {}".format(distance), **kwargs)

    return job


# In[5]:


# Funtion for getting the required Gaia data to construct HR Diagram

# the query runs for a while, better ensure we have those data saved

try:
    log.info("Getting the DR2 results on nearby (d < 200 pc) stars...")
    gaiarec = np.recfromcsv("gaia-hrd-dr2-200pc.csv")
    bp_rp, mg = gaiarec.bp_rp, gaiarec.mg
    
except OSError:
    job = send_radial_gaia_query(dump_to_file=True, output_format="csv",
                                 output_file="gaia-hrd-dr2-200pc.csv",
                                 query_size=10000000)
    r = job.get_results()
    bp_rp = r['bp_rp'].data
    mg = r['mg'].data


# In[6]:


# Jan's query code

def Xmatch_PS(ra,dec,max_distance=5.):
    # make a python table with coordinates
    mycat = Table([ra,dec],names=['myra','mydec'],
        meta={'name':'mytable'})

    # download data from vizier
    print('Downloading PS data')

    # download
    table = XMatch.query(cat1=mycat,
           cat2='vizier:II/349/ps1',
           max_distance=max_distance * u.arcsec,
           colRA1='myra',
           colDec1='mydec')

    # vizier returns everything close to a position, so we need to
    # to find only the nearest star in the NOMAD data
    c1 = SkyCoord(ra*u.deg,dec*u.deg, frame='icrs')
    c2 = SkyCoord(table['RAJ2000']*u.deg,table['DEJ2000']*u.deg,
    frame='icrs')
    idx, d2d, d3d = c1.match_to_catalog_sky(c2)

    # make the output
    output = [np.array(table[colname].tolist(),dtype=float) for colname in
             ['gmag','e_gmag','rmag','e_rmag','imag','e_imag','zmag','e_zmag','ymag','e_ymag']]
    output = np.column_stack(output)[idx]
    m = d2d>max_distance*u.arcsec
    output[m] = np.nan*np.ones(10)

    return output


def Xmatch_SDSS(ra,dec,max_distance=5.):
    # make a python table with coordinates
    mycat = Table([ra,dec],names=['myra','mydec'],
        meta={'name':'mytable'})

    # download data from vizier
    print('Downloading SDSS data')

    # download
    table = XMatch.query(cat1=mycat,
           cat2='vizier:V/147/sdss12',
           max_distance=max_distance * u.arcsec,
           colRA1='myra',
           colDec1='mydec')

    print(table)
   
    # vizier returns everything close to a position, so we need to
    # to find only the nearest star in the NOMAD data
    c1 = SkyCoord(ra*u.deg,dec*u.deg, frame='icrs')
    c2 = SkyCoord(table['RAdeg']*u.deg,table['DEdeg']*u.deg,
    frame='icrs')
    idx, d2d, d3d = c1.match_to_catalog_sky(c2)

    # make the output
    output = [np.array(table[colname].tolist(),dtype=float) for colname in
             ['umag','e_umag','gmag','e_gmag','rmag','e_rmag','imag','e_imag','zmag','e_zmag']]
    output = np.column_stack(output)[idx]
    m = d2d>max_distance*u.arcsec
    output[m] = np.nan*np.ones(10)

    return output


def Xmatch_Gaia(ra,dec,max_distance=5.):
    # make a python table with coordinates
    mycat = Table([ra,dec],names=['myra','mydec'],
        meta={'name':'mytable'})

    # download data from vizier
    print('Downloading Gaia data')

    # download
    table = XMatch.query(cat1=mycat,
           cat2='vizier:I/345/gaia2',
           max_distance=max_distance * u.arcsec,
           colRA1='myra',
           colDec1='mydec')

    # vizier returns everything close to a position, so we need to
    # to find only the nearest star in the NOMAD data
    c1 = SkyCoord(ra*u.deg,dec*u.deg, frame='icrs')
    c2 = SkyCoord(table['ra_epoch2000']*u.deg,table['dec_epoch2000']*u.deg,
        frame='icrs')
    idx, d2d, d3d = c1.match_to_catalog_sky(c2)

    # make the output
    output = [np.array(table[colname].tolist(),dtype=float) for colname in
             ['parallax','parallax_error',
              'pmra','pmra_error','pmdec','pmdec_error', 
              'phot_g_mean_mag',
              'phot_bp_mean_mag','phot_rp_mean_mag']]
    output = np.column_stack(output)[idx]
    m = d2d>max_distance*u.arcsec
    output[m] = np.nan*np.ones(9)

    return output


def Xmatch_Gaia_edr3(ra,dec,max_distance=5.):
    # make a python table with coordinates
    mycat = Table([ra,dec],names=['myra','mydec'],
        meta={'name':'mytable'})

    # download data from vizier
    print('Downloading Gaia data')

    # download
    table = XMatch.query(cat1=mycat,
           cat2='vizier:I/350/gaiaedr3',
           max_distance=max_distance * u.arcsec,
           colRA1='myra',
           colDec1='mydec')

    # vizier returns everything close to a position, so we need to
    # to find only the nearest star in the NOMAD data
    c1 = SkyCoord(ra*u.deg,dec*u.deg, frame='icrs')
    c2 = SkyCoord(table['ra_epoch2000']*u.deg,table['dec_epoch2000']*u.deg,
        frame='icrs')
    idx, d2d, d3d = c1.match_to_catalog_sky(c2)

    # make the output
    output = [np.array(table[colname].tolist(),dtype=float) for colname in
             ['parallax','parallax_error',
              'pmra','pmra_error','pmdec','pmdec_error', 
              'phot_g_mean_mag',
              'phot_bp_mean_mag','phot_rp_mean_mag']]
    output = np.column_stack(output)[idx]
    m = d2d>max_distance*u.arcsec
    output[m] = np.nan*np.ones(9)

    return output



def Xmatch_Gaia_dist(ra,dec,max_distance=5.):
    # make a python table with coordinates
    mycat = Table([ra,dec],names=['myra','mydec'],
        meta={'name':'mytable'})

    # download data from vizier
    print('Downloading Gaia dist data')

    # download
    table = XMatch.query(cat1=mycat,
           cat2='vizier:I/347/gaia2dis',
           max_distance=max_distance * u.arcsec,
           colRA1='myra',
           colDec1='mydec')

    # vizier returns everything close to a position, so we need to
    # to find only the nearest star in the NOMAD data
    c1 = SkyCoord(ra*u.deg,dec*u.deg, frame='icrs')
    c2 = SkyCoord(table['RA_ICRS']*u.deg,table['DE_ICRS']*u.deg,
        frame='icrs')
    idx, d2d, d3d = c1.match_to_catalog_sky(c2)

    # make the output
    output = [np.array(table[colname].tolist(),dtype=float) for colname in
             ['rest','b_rest','B_rest']]
    output = np.column_stack(output)[idx]
    m = d2d>max_distance*u.arcsec
    output[m] = np.nan*np.ones(3)

    return output

def Xmatch_Gaia_dist_edr3(ra,dec,max_distance=5.):
    # make a python table with coordinates
    mycat = Table([ra,dec],names=['myra','mydec'],
        meta={'name':'mytable'})

    # download data from vizier
    print('Downloading Gaia dist data')

    # download
    table = XMatch.query(cat1=mycat,
           cat2='vizier:I/352/gaia3dis',
           max_distance=max_distance * u.arcsec,
           colRA1='myra',
           colDec1='mydec')

    # vizier returns everything close to a position, so we need to
    # to find only the nearest star in the NOMAD data
    c1 = SkyCoord(ra*u.deg,dec*u.deg, frame='icrs')
    c2 = SkyCoord(table['RA_ICRS']*u.deg,table['DE_ICRS']*u.deg,
        frame='icrs')
    idx, d2d, d3d = c1.match_to_catalog_sky(c2)

    # make the output
    output = [np.array(table[colname].tolist(),dtype=float) for colname in
             ['rest','b_rest','B_rest']]
    output = np.column_stack(output)[idx]
    m = d2d>max_distance*u.arcsec
    output[m] = np.nan*np.ones(3)

    return output


def Xmatch_GALEX(ra,dec,max_distance=5.):

    # make a python table with coordinates
    mycat = Table([ra,dec],names=['myra','mydec'],
        meta={'name':'mytable'})

    # download data from vizier
    print('Downloading GALEX data')

    # download
    table = XMatch.query(cat1=mycat,
           cat2='vizier:II/312/ais',
           #cat2='vizier:II/335/galex_ais',
           max_distance=max_distance * u.arcsec,
           colRA1='myra',
           colDec1='mydec')
           #colRA2='RAJ2000',
           #colDec2='DEJ2000')

    #print(table)
    print(table)
    print(len(table))
    if len(table)==0:
        return  np.nan*np.ones([len(ra),4])

    # vizier returns everything close to a position, so we need to
    # to find only the nearest star in the NOMAD data
    c1 = SkyCoord(ra*u.deg,dec*u.deg, frame='icrs')
    c2 = SkyCoord(table['ra']*u.deg,table['dec']*u.deg, frame='icrs')
    idx, d2d, d3d = c1.match_to_catalog_sky(c2)

    # make the output
    #output = [np.array(table[colname].tolist(),dtype=float) for colname in
    #         ['NUVmag','e_NUVmag','FUVmag','e_FUVmag']]
    output = [np.array(table[colname].tolist(),dtype=float) for colname in
             ['fuv_mag','fuv_magerr','nuv_mag','nuv_magerr',]]
    output = np.column_stack(output)[idx]
    m = d2d>max_distance*u.arcsec
    output[m] = np.nan*np.ones(4)

    return output


# In[7]:


def LombScargle_MultibandFast(file_name, data, sigma_lower = 3, sigma_upper = 3, period_LowerLimit = 0.04, period_UpperLimit = 0.7):
    
    # Input parameters:
    
        # file_name = name of the light curve file
        
        # data = light curve file
            
        # sigma_lower = the number of standard deviations away from the median to use as the lower bound for the sigma_clip limit
            # Default is 3
            
        # sigma_upper = the number of standard deviations away from the median to use as the upper bound for the sigma_clip limit
            # Default is 3
            
        # period_LowerLimit = lowest limit of the period when using Lomb-Scargle
            # Default is 0.04 days = 57.6 minutes
            
        # period_UpperLimit = Highest limit of the period when using Lomb-Scargle
            # Default is 0.7 days = 1008 minutes
    
    # If there are less than 50 data points in total
    if data.size < 50:
        
        print("\n")
        
        # Return 0 for everything besides the file name
        return([file_name, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    
    # Calculate the number of data points with flags == 0
    flags =  (data[:, 8])
    flags_num = len(flags[flags == 0])  
    
    # If the number of data points with flags == 0 is less than 50
    if flags_num < 50:

        print("\n")

        # Return 0 for everything besides the file name
        return([file_name, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
           
    # Store the ra and dec info by reading the name of the file
    index = file_name.find("_", 5)
    ra_deg = float(file_name[3:index])
    dec_deg = float(file_name[index+1:len(file_name)-4])
    
    # Create a SkyCoord object
    c = SkyCoord(ra_deg, dec_deg, frame = 'icrs', unit = "deg") 

    # Coordinate Conversion from degree to ICRS
    coord = c.to_string('hmsdms', precision = 1)

    # Separate the coordinates into ra and dec
    ra_icrs = coord[0:11]
    dec_icrs = coord[12:]

    # Load in the values that have flags == 0 
    data = data[flags == 0]

    # t = Heliocentric Julian Date            
    t =  data[:, 0]
    # flux = flux
    flux = data[:, 1]
    # dflux = error
    dflux = data[:, 2]
    # filts = filter id
    filts = data[:, 3]
    
    # Make a mask for the NaN values in array t
    mask_nan = (np.isnan(t) == True) 
    
    # Remove the data that have the NaN values
    t = t[mask_nan == False]
    flux = flux[mask_nan == False]
    dflux = dflux[mask_nan == False]
    filts = filts[mask_nan == False]
    
    # Make a mask for the non-positive values in the array flux
    mask_flux = (flux <= 0) 
    
    # Remove the data that have the non-positive values
    t = t[mask_flux == False]
    flux = flux[mask_flux == False]
    dflux = dflux[mask_flux == False]
    filts = filts[mask_flux == False]
    
    # Make a mask for the non-positive values in the array dflux
    mask_dflux = (dflux <= 0) 
    
    # Remove the data that have the non-positive values
    t = t[mask_dflux == False]
    flux = flux[mask_dflux == False]
    dflux = dflux[mask_dflux == False]
    filts = filts[mask_dflux == False]
    
    # Create some empty arrays for later use

    flux_g = []
    flux_r = []

    t_c = []
    flux_c = []
    dlfux_c = []
    filts_c = []

    clipped_g = []
    clipped_r = []

    flux_gp_r = []
    flux_gp_g = []
    
    sigma_gp_r = []
    sigma_gp_g = []
    
    Gaia_crossmatch = []
    
    # Set up the kernel
    kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1, 1e2)) + W(noise_level = 1e-3, noise_level_bounds = (1e-06, 1))
    
    # If there is no data in the g filter and there are more than 25 data points in the r filter
    if(np.size(flux[filts == 1]) == 0) and (np.size(flux[filts == 2]) > 25):
        
        # Divide flux and dflux by the medians
        flux_r_divide_by_median = flux[filts == 2] / np.nanmedian(flux[filts == 2])
        dflux_r_divide_by_median = dflux[filts == 2] / np.nanmedian(dflux[filts == 2])
        
        # Converts flux of the r data to magnitude
        mag_r = flux2mag(flux[filts == 2])

        # Calculate the mean magnitude of the r data with 2 decimals
        mag_r_avg = round(np.nanmean(mag_r), 2)

        # Set the mean magnitude of the g data to -999
        mag_g_avg = -999

        # Convert inputs to arrays with at least two dimension
        t_2d_r = np.atleast_2d(t[filts == 2]).T

        # Instantiate a Gaussian Process model
        gp_r = GaussianProcessRegressor(kernel = kernel, alpha = dflux_r_divide_by_median ** 2, n_restarts_optimizer = 3, normalize_y = True)
        
        # Fit to data using Maximum Likelihood Estimation of the parameters
        gp_r.fit(t_2d_r, flux_r_divide_by_median)

        # Mesh the input space for evaluations of the real function, the prediction and its MSE
        x_r = np.atleast_2d(np.linspace(np.nanmin(t[filts == 2]), np.nanmax(t[filts == 2]), 1000)).T

        # Make the prediction on the meshed x-axis (ask for MSE as well)
        y_pred_r, sigma_r = gp_r.predict(x_r, return_std = True)

        # Make the prediction on the original t values
        flux_gp_r, sigma_gp_r = gp_r.predict(t_2d_r, return_std = True)

        # Run sigma_clip on the r data 
        clipped_r = sigma_clip(flux_r_divide_by_median / flux_gp_r, sigma_lower = sigma_lower, sigma_upper = sigma_upper, maxiters = 5, cenfunc = np.median, masked = True, copy = True)

        # Mask the r data
        t_r, flux_r, dflux_r, filts_r = (t[filts == 2])[~clipped_r.mask], flux_r_divide_by_median[~clipped_r.mask], dflux_r_divide_by_median[~clipped_r.mask], (filts[filts == 2])[~clipped_r.mask] 

        # Set the number of the clipped g data points to 0
        clipped_number_g = 0

        # Calculate the number of clipped data points in the r filter
        clipped_number_r = len(t[filts == 2])-len(t_r)

        # Store values in the empty arrays created before
        t_c = t_r
        flux_c = flux_r / flux_gp_r[~clipped_r.mask]
        dflux_c = dflux_r / flux_gp_r[~clipped_r.mask]
        filts_c = filts_r

    # If there is no data in the r filter and there are more than 25 data points in the g filter
    elif (np.size(flux[filts == 2]) == 0) and (np.size(flux[filts == 1]) > 25):
        
        # Divide flux and dflux by the medians
        flux_g_divide_by_median = flux[filts == 1] / np.nanmedian(flux[filts == 1])
        dflux_g_divide_by_median = dflux[filts == 1] / np.nanmedian(dflux[filts == 1])

        # Converts flux of the g data to magnitude
        mag_g = flux2mag(flux[filts == 1])

        # Calculate the mean magnitude of the g data with 2 decimals
        mag_g_avg = round(np.nanmean(mag_g), 2)

        # Set the mean magnitude of the r data to -999
        mag_r_avg = -999

        # Convert inputs to arrays with at least two dimension
        t_2d_g = np.atleast_2d(t[filts == 1]).T

        # Instantiate a Gaussian Process model
        gp_g = GaussianProcessRegressor(kernel = kernel, alpha = dflux_g_divide_by_median ** 2, n_restarts_optimizer = 3, normalize_y = True)

        # Fit to data using Maximum Likelihood Estimation of the parameters
        gp_g.fit(t_2d_g, flux_g_divide_by_median)

        # Mesh the input space for evaluations of the real function, the prediction and its MSE
        x_g = np.atleast_2d(np.linspace(np.nanmin(t[filts == 1]), np.nanmax(t[filts == 1]), 1000)).T

        # Make the prediction on the meshed x-axis (ask for MSE as well)
        y_pred_g, sigma_g = gp_g.predict(x_g, return_std = True)

        # Make the prediction on the original t values
        flux_gp_g, sigma_gp_g = gp_g.predict(t_2d_g, return_std=True)

        # Run sigma_clip on the g data  
        clipped_g = sigma_clip(flux_g_divide_by_median / flux_gp_g, sigma_lower = sigma_lower, sigma_upper = sigma_upper, maxiters = 5, cenfunc = np.median, masked = True, copy = True)

        # Mask the g data
        t_g, flux_g, dflux_g, filts_g = (t[filts == 1])[~clipped_g.mask], flux_g_divide_by_median[~clipped_g.mask], dflux_g_divide_by_median[~clipped_g.mask], (filts[filts == 1])[~clipped_g.mask] 

        # Set the number of the clipped r data points to 0
        clipped_number_r = 0

        # Calculate the number of clipped data points in the g filter
        clipped_number_g = len(t[(filts == 1)])-len(t_g)

        # Store values in the empty arrays created before
        t_c = t_g
        flux_c = flux_g / flux_gp_g[~clipped_g.mask]
        dflux_c = dflux_g / flux_gp_g[~clipped_g.mask]
        filts_c = filts_g

    # If there are more than 25 data points in both the g filter and r filter
    elif (np.size(flux[filts == 1]) > 25) and (np.size(flux[filts == 2]) > 25):
        
        # Divide flux and dflux by the medians

        # g filter
        flux_g_divide_by_median = flux[filts == 1] / np.nanmedian(flux[filts == 1])
        dflux_g_divide_by_median = dflux[filts == 1] / np.nanmedian(dflux[filts == 1])

        # r filter
        flux_r_divide_by_median = flux[filts == 2] / np.nanmedian(flux[filts == 2])
        dflux_r_divide_by_median = dflux[filts == 2] / np.nanmedian(dflux[filts == 2])

        # Converts flux to magnitude
        mag_g = flux2mag(flux[(filts == 1)])
        mag_r = flux2mag(flux[(filts == 2)])

        # Calculate the mean magnitude with 2 decimals
        mag_g_avg = round(np.nanmean(mag_g), 2)
        mag_r_avg = round(np.nanmean(mag_r), 2)

        # Convert inputs to arrays with at least two dimension
        t_2d_g = np.atleast_2d(t[filts == 1]).T
        t_2d_r = np.atleast_2d(t[filts == 2]).T

        # Gaussian Processes: g filter 
        # Instantiate a Gaussian Process model
        gp_g = GaussianProcessRegressor(kernel = kernel, alpha = dflux_g_divide_by_median ** 2, n_restarts_optimizer = 3, normalize_y = True)
        
        # Fit to data using Maximum Likelihood Estimation of the parameters
        gp_g.fit(t_2d_g, flux_g_divide_by_median)
        
        # Mesh the input space for evaluations of the real function, the prediction and its MSE
        x_g = np.atleast_2d(np.linspace(np.nanmin(t[filts == 1]), np.nanmax(t[filts == 1]), 1000)).T
        
        # Make the prediction on the meshed x-axis (ask for MSE as well)
        y_pred_g, sigma_g = gp_g.predict(x_g, return_std = True)
        
        # Make the prediction on the original t values
        flux_gp_g, sigma_gp_g = gp_g.predict(t_2d_g, return_std = True)

        # Gaussian Processes: r filter 
        # Instantiate a Gaussian Process model
        gp_r = GaussianProcessRegressor(kernel = kernel, alpha = dflux_r_divide_by_median ** 2, n_restarts_optimizer = 3, normalize_y = True)

        # Fit to data using Maximum Likelihood Estimation of the parameters
        gp_r.fit(t_2d_r, flux_r_divide_by_median)

        # Mesh the input space for evaluations of the real function, the prediction and its MSE
        x_r = np.atleast_2d(np.linspace(np.nanmin(t[filts == 2]), np.nanmax(t[filts == 2]), 1000)).T
        
        # Make the prediction on the meshed x-axis (ask for MSE as well)
        y_pred_r, sigma_r = gp_r.predict(x_r, return_std = True)
        
        # Make the prediction on the original t values
        flux_gp_r, sigma_gp_r = gp_r.predict(t_2d_r, return_std = True)

        # Run sigma_clip on the g and r data separately 
        clipped_g = sigma_clip(flux_g_divide_by_median / flux_gp_g, sigma_lower = sigma_lower, sigma_upper = sigma_upper, maxiters = 5, cenfunc = np.median, masked = True, copy = True)
        clipped_r = sigma_clip(flux_r_divide_by_median / flux_gp_r, sigma_lower = sigma_lower, sigma_upper = sigma_upper, maxiters = 5, cenfunc = np.median, masked = True, copy = True)

        # Mask the g and r data
        t_g, flux_g, dflux_g, filts_g = (t[filts == 1])[~clipped_g.mask], flux_g_divide_by_median[~clipped_g.mask], dflux_g_divide_by_median[~clipped_g.mask], (filts[filts == 1])[~clipped_g.mask] 
        t_r, flux_r, dflux_r, filts_r = (t[filts == 2])[~clipped_r.mask], flux_r_divide_by_median[~clipped_r.mask], dflux_r_divide_by_median[~clipped_r.mask], (filts[filts == 2])[~clipped_r.mask] 

        # Calculate the number of clipped data points
        clipped_number_g = len(t[(filts == 1)])-len(t_g)
        clipped_number_r = len(t[(filts == 2)])-len(t_r)

        # Store the clipped g and r data in the arrays created before
        t_c = np.concatenate((t_g, t_r), axis=0)
        flux_c = np.concatenate((flux_g / flux_gp_g[~clipped_g.mask], flux_r / flux_gp_r[~clipped_r.mask]), axis=0)
        dflux_c = np.concatenate((dflux_g / flux_gp_g[~clipped_g.mask], dflux_r / flux_gp_r[~clipped_r.mask]), axis=0)
        filts_c = np.concatenate((filts_g, filts_r), axis=0)

    else: 

        print("\n")
        
        # Return 0 for everything besides the file name
        return([file_name, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    # Use LombScargleMultibandFast
    model = periodic.LombScargleMultibandFast(fit_period = False)
    # Set the period range [days]
    model.optimizer.period_range = (period_LowerLimit, period_UpperLimit)
    # Run the fit
    model.fit(t_c, flux_c, dflux_c, filts_c);

    # Convert the period from days to minutes
    period_min = model.best_period * 24 * 60

    # Calculate the number of steps
    baseline = np.max(t) - np.min(t)
    df = 1. / baseline
    f_min = 1. / (model.best_period * 2.5)
    f_max = 1. / period_LowerLimit
    nf = int(np.ceil(f_max - f_min) / df)
    
    # Calculate the parameters for the periodogram
    periods = np.linspace(period_LowerLimit, model.best_period * 2.5, nf * 3)
    P_multi = model.periodogram(periods)

    # Compute the highest power and the significance 
    highest_power = np.nanmax(P_multi)
    ratio = (np.nanmax(P_multi)-np.nanmedian(P_multi))/np.nanstd(P_multi)
    

    

#####################################################################################################################################################
#####################################################################################################################################################
#####################################################################################################################################################
#####################################################################################################################################################
#####################################################################################################################################################


    # Create a subplot 
    f = plt.figure(figsize = (16,12))

    # 1st Plot: Clipped Data
    ax1 = f.add_subplot(3,2,(1,2)) 

    # If the number of g data point is not 0:
    if(np.size(flux[(filts == 1)]) != 0):

        # Plot the unclipped g data
        ax1.scatter(t[filts == 1][~clipped_g.mask], flux2mag(flux[filts == 1][~clipped_g.mask]), color = 'turquoise', marker = '.', label = 'unclipped g data')

        # Plot the clipped g data
        ax1.scatter(t[filts == 1][clipped_g.mask], flux2mag(flux[filts == 1][clipped_g.mask]), color = 'g', marker = 'x', label = 'clipped g data')

    # If the number of r data point is not 0:
    if(np.size(flux[(filts == 2)]) != 0):

        # Plot the unclipped r data
        ax1.scatter(t[filts == 2][~clipped_r.mask], flux2mag(flux[filts == 2][~clipped_r.mask]), color = 'lightcoral', marker = '.', label = 'unclipped r data')

        # Plot the clipped r data
        ax1.scatter(t[filts == 2][clipped_r.mask], flux2mag(flux[filts == 2][clipped_r.mask]), color = 'r', marker = 'x', label = 'clipped r data')

    # Set the x and y labels
    ax1.set(xlabel='Heliocentric Julian Date', ylabel='Magnitude')
    # Turn off scientific notation 
    ax1.ticklabel_format(style='plain')
    # Invert the y axis
    ax1.invert_yaxis()
    # Set the legend
    ax1.legend(bbox_to_anchor=(1.0175, 1.0), loc = 'upper left')
    # Set the title
    ax1.set_title("Light Curve")

    # 2nd Plot: Gaussian Processes
    ax2 = f.add_subplot(3,2,3) 

    # If the number of g data point is not 0:
    if(np.size(flux[filts == 1]) != 0):

        # Plot the observations
        ax2.scatter(t[filts == 1], flux_g_divide_by_median, color = 'turquoise', s = 4, label='g data')

        # Plot the Prediction
        ax2.plot(x_g, y_pred_g, color = 'green', zorder = 5, label='g prediction')

        # Plot the 95% confidence interval
        ax2.fill(np.concatenate([x_g, x_g[::-1]]), 
                 np.concatenate([y_pred_g - 1.9600 * sigma_g, (y_pred_g + 1.9600 * sigma_g)[::-1]]),
                 alpha = 0.5, fc = 'green', ec = 'None', label = 'g prediction 95% confidence interval')

    # If the number of r data point is not 0:
    if(np.size(flux[filts == 2]) != 0):

        # Plot the observations
        ax2.scatter(t[filts == 2], flux_r_divide_by_median, color = 'lightcoral', s = 4, label='r data')

        # Plot the Prediction
        ax2.plot(x_r, y_pred_r, color = 'red', zorder = 5, label='r prediction')

        # Plot the 95% confidence interval
        ax2.fill(np.concatenate([x_r, x_r[::-1]]), 
                 np.concatenate([y_pred_r - 1.9600 * sigma_r, (y_pred_r + 1.9600 * sigma_r)[::-1]]), 
                 alpha = 0.5, fc = 'red', ec = 'None', label = 'r prediction 95% confidence interval')

    # Set the x and y labels
    ax2.set(xlabel = 'Heliocentric Julian Date', ylabel = 'Flux/Median')
    # Turn off scientific notation 
    ax2.ticklabel_format(style='plain')
    # Set the legend
    ax2.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    # Set the title
    ax2.set_title("Gaussian Process Regression")

    # 3rd Plot: "Plot of Flux with Best-Fit Period"
    ax3 = f.add_subplot(3,2,4) 

    # Create 1000 x values for the fit 
    tfit = np.linspace(0, model.best_period, 1000)

    # Predict on a regular phase grid
    phase = (t_c / model.best_period) % 1
    phasefit = (tfit / model.best_period)

    # If the number of g data point is not 0:
    if(np.size(flux_c[(filts_c == 1)]) != 0):

        # Calculate the fit values for the 1(G) filter
        fluxfit_g = model.predict(tfit, filts = 1)

        # Plot the error bars for the G data 
        #ax3.errorbar(phase[(filts_c == 1)], flux_c[(filts_c == 1)], dflux_g, fmt = '.', color = 'turquoise', linewidth = 1, label = 'g data')
        ax3.scatter(phase[(filts_c == 1)], flux_g / flux_gp_g[~clipped_g.mask], color = 'turquoise', s = 10, label = 'g data')
        
        # Plot the fit for the G data 
        ax3.plot(phasefit, fluxfit_g, color = 'g', zorder = 20, linewidth = 2, label = 'g fit')

        # Plot phase + 1
        
        # Plot the error bars for the G data 
        #ax3.errorbar(phase[(filts_c == 1)] + 1, flux_c[(filts_c == 1)], dflux_g, fmt = '.', color = 'turquoise', linewidth = 1)
        ax3.scatter(phase[(filts_c == 1)] + 1, flux_g / flux_gp_g[~clipped_g.mask], color = 'turquoise', s = 10)
        
        # Plot the fit for the G data 
        ax3.plot(phasefit+1, fluxfit_g, color = 'g', zorder = 20, linewidth = 2)


    # If the number of r data point is not 0:
    if(np.size(flux_c[(filts_c == 2)]) != 0):    

        # Calculate the fit values for the 2(R) filter
        fluxfit_r = model.predict(tfit, filts = 2)

        # Plot the error bars for the R data 
        #ax3.errorbar(phase[(filts_c == 2)], flux_c[(filts_c == 2)], dflux_r, fmt = '.', color = 'lightcoral', linewidth = 1, label = 'r data')
        ax3.scatter(phase[(filts_c == 2)], flux_r / flux_gp_r[~clipped_r.mask], color = 'lightcoral', s = 10, label = 'r data')
        
        # Plot the fit for the R data 
        ax3.plot(phasefit, fluxfit_r, color = 'r', zorder = 20, linewidth = 2, label = 'r fit')

        # Plot phase + 1
        
        # Plot the error bars for the R data 
        #ax3.errorbar(phase[(filts_c == 2)] + 1, flux_c[(filts_c == 2)], dflux_r, fmt = '.', color = 'lightcoral', linewidth = 1)
        ax3.scatter(phase[(filts_c == 2)] + 1, flux_r / flux_gp_r[~clipped_r.mask], color = 'lightcoral', s = 10)
        
        # Plot the fit for the R data 
        ax3.plot(phasefit+1, fluxfit_r, color = 'r', zorder = 20, linewidth = 2)

    if flux_gp_g == []:
        pos = np.nanmax(flux_r / flux_gp_r[~clipped_r.mask])
    elif flux_gp_r == []:
        pos = np.nanmax(flux_g / flux_gp_g[~clipped_g.mask])
    elif np.nanmax(flux_g / flux_gp_g[~clipped_g.mask]) > np.nanmax(flux_r / flux_gp_r[~clipped_r.mask]):
        pos = np.nanmax(flux_g / flux_gp_g[~clipped_g.mask])
    else: 
        pos = np.nanmax(flux_r / flux_gp_r[~clipped_r.mask])

    # Add text to indicate the best period
    ax3.text(0, pos, "The best period is {:.3f} days = {:.0f} minutes".format(model.best_period, period_min))
    # Set the labels for the x and y axes
    ax3.set(xlabel='Phase', ylabel = 'Normalized Flux')
    # Set the y limit
    #ax3.set_ylim(bottom = None, top = pos * 1.05)
    # Set the legend
    ax3.legend(bbox_to_anchor=(1.05, 1.0), loc = 'upper left')
    # Set the title
    ax3.set_title("Best-Fit Period")

    # 4th Plot: "Multiband Periodogram"
    ax4 = f.add_subplot(3,2,5) 

    # Make and plot the Periodogram
    ax4.plot(periods, P_multi, lw = 1, color = 'steelblue')

    # Mark the period with the highest power
    ax4.scatter(periods[np.nanargmax(P_multi)], np.nanmax(P_multi), color = 'r', marker = 'x', linewidth = 2, label = 'Highest power')

    # Output the information on the Periodogram plot
    ax4.text(period_LowerLimit, np.nanmax(P_multi) * 1.22, "    The highest Lomb-Scargle power is {:.3f}".format(np.nanmax(P_multi)))
    ax4.text(period_LowerLimit, np.nanmax(P_multi) * 1.12, "    (highest power-median)/std = {:.3f}".format(ratio))

    # Set the title
    ax4.set_title('Multiband Periodogram', fontsize=12)
    # Set the x-axis label
    ax4.set_xlabel('Period (days)')
    # Make the x axis logarithmic
    #ax4.set_xscale('log')
    # Show the x values and turn off the scientific notation
    ax4.tick_params(axis='x', which='minor')
    ax4.xaxis.set_minor_formatter(FormatStrFormatter("%.2f"))
    # Set the y-axis label
    ax4.set_ylabel('Power')
    # Set the y limit
    ax4.set_ylim(0, np.nanmax(P_multi) * 1.3)
    # Set the legend
    ax4.legend(bbox_to_anchor=(1.0175, 1.0), loc = 'upper left')

    # 5th Plot: "Multiband Periodogram"
    ax5 = f.add_subplot(3,2,6) 

    # only show 2D-histogram for bins with more than 10 stars in them
    h = ax5.hist2d(bp_rp, mg, bins=300, cmin=10, norm=colors.PowerNorm(0.5), zorder=0.5)
    # fill the rest with scatter (set rasterized=True if saving as vector graphics)
    ax5.scatter(bp_rp, mg, alpha=0.05, s=1, color='k', zorder=0)

    # Gaia Xmatch
    try:
        Gaia_crossmatch = Xmatch_Gaia([ra_deg], [dec_deg], 5)
        

        parallax = Gaia_crossmatch[0][0]
        parallax_error = Gaia_crossmatch[0][1]
        g_mean_mag = Gaia_crossmatch[0][6]
        gaia_bp = Gaia_crossmatch[0][7]
        gaia_rp = Gaia_crossmatch[0][8]

        # Calculate the absolute magnitude
        abs_mag_g, error_g = absolute_magnitude(g_mean_mag, parallax, parallax_error)

        # Calculate the color
        color = gaia_bp - gaia_rp

        # Plot the point on the HR Diagram
        ax5.errorbar(color, abs_mag_g, error_g, color = 'red', fmt = 'o', lw = 0.5)
        
    except:
        g_mean_mag = -999
        gaia_bp = -999
        gaia_rp = -999
        color = -999
        
    ax5.invert_yaxis()
    cb = f.colorbar(h[3], pad=0.02)
    ax5.set_xlabel(r'$G_{BP} - G_{RP}$')
    ax5.set_ylabel(r'$M_G$')
    ax5.set_title("Hertzsprung-Russell Diagram")
    cb.set_label(r"$\mathrm{Stellar~Density}$")

    # Name of the title 
    title = str(ra_deg) + "_" + str(dec_deg) + " (" + ra_icrs + "_" + dec_icrs + ")" 

    # Set the file name as the suptitle
    plt.suptitle(title, y = 0.99, horizontalalignment='center')

    plt.tight_layout()

    # Name of the subplot 
    plot_name = "Plots/" + str(ra_deg) + "_" + str(dec_deg) + " (" + ra_icrs + "_" + dec_icrs + ")" + ".png"

    # Save the subplot
    plt.savefig(plot_name, dpi = 200)

    # If the period is between 60 and 120 minutes with a significance >= 8.5, save the plot in the "Plots_Short_Period_Likely" folder
    if((period_min >= 60) & (period_min <= 120) & (ratio >= 8.5)):
        # Name of the subplot 
        plot_name = "Plots_Short_Period_Likely/" + file_name + ".png"
        # Save the plot
        plt.savefig(plot_name, dpi = 200)

    # If the period is between 60 and 120 minutes with a significance < 8.5, save the plot in the "Plots_Short_Period_Possible" folder
    elif((period_min >= 60) & (period_min <= 120) & (ratio < 8.5)):
        # Name of the subplot 
        plot_name = "Plots_Short_Period_Possible/" + file_name + ".png"
        # Save the plot
        plt.savefig(plot_name, dpi = 200)

    # Close the plot
    plt.close()    

    print("\n")

    # Return the best period in days and minutes, the highest Lomb-Scargle Power, the ratio of the highest power to the median,
    # the average magnitude, and the number of data points
    return([file_name, ra_deg, dec_deg, ra_icrs, dec_icrs, mag_g_avg, mag_r_avg, g_mean_mag, gaia_bp, gaia_rp, color, flags_num, clipped_number_g, clipped_number_r, round(model.best_period,3), round(period_min*10/10,0), round(highest_power,3), 
            round(ratio,3)])



#####################################################################################################################################################
#####################################################################################################################################################
#####################################################################################################################################################
#####################################################################################################################################################
#####################################################################################################################################################

# Store the names of the files in an array
files = glob.glob('lc*.dat')

# Names of the output files
output_file = "Output.txt"
output_short_likely_file = "Output_Short_Period_Likely.txt"
output_short_possible_file = "Output_Short_Period_Possible.txt"


# If Output.txt exists
if path.isfile('Output.txt') == True:

    # Load in Output.txt
    info = np.loadtxt('Output.txt', skiprows = 1, dtype = str)

    
# Loop through each light curve file
for i in range(len(files)):
    
    print(files[i])

    # If "Output.txt" exists
    if path.isfile('Output.txt') == True:
    
        # If info is a 1D array
        if len(info.shape) == 1:

            # Store the names of the recorded light curves in Output.txt
            name = info[0]

        # If info is a 2D array
        if len(info.shape) == 2:

            # Store the names of the recorded light curves in Output.txt
            name = info[:,0]

        # If the name of the light curve is not recorded in Output.txt, run the LombScargle_MultibandFast function
        if str(files[i]) not in name:

            # Load in the data on the light curve file
            data = np.loadtxt(files[i])

            # Store the outputs of the LombScargle_MultibandFast function
            output = LombScargle_MultibandFast(files[i], data)

            # If the period is between 60 and 120 minutes with a significance >= 8.5
            if((output[15] >= 60) & (output[15] <= 120) & (output[17] >= 8.5)): 

                if not path.exists(output_short_likely_file):
                    with open(output_short_likely_file,'a+') as myfile:
                        myfile.write("File_Name, RA(degree), DEC(degree), RA(ICRS), DEC(ICRS), Average_G_Magnitude, Average_R_Magnitude, Gaia_G, Gaia_bp, Gaia_rp, Color, Data_Points, #Clipped_G, #Clipped_R, Best_Period(days), Best_Period(minutes), Highest_Lomb-Scargle_Power, (HighestPower-Median)/std" + "\n")

                with open(output_short_likely_file, "a+") as myfile: 
                    myfile.write(' '.join(str(e) for e in output) + "\n")      

            # If the period is between 60 and 120 minutes with a significance < 8.5
            if((output[15] >= 60) & (output[15] <= 120) & (output[17] < 8.5)): 

                if not path.exists(output_short_possible_file):
                    with open(output_short_possible_file,'a+') as myfile:
                        myfile.write("File_Name, RA(degree), DEC(degree), RA(ICRS), DEC(ICRS), Average_G_Magnitude, Average_R_Magnitude, Gaia_G, Gaia_bp, Gaia_rp, Color, Data_Points, #Clipped_G, #Clipped_R, Best_Period(days), Best_Period(minutes), Highest_Lomb-Scargle_Power, (HighestPower-Median)/std" + "\n")

                with open(output_short_possible_file, "a+") as myfile: 
                    myfile.write(' '.join(str(e) for e in output) + "\n")  
                    
            if not path.exists(output_file):
                with open(output_file,'a+') as myfile:
                    myfile.write("File_Name, RA(degree), DEC(degree), RA(ICRS), DEC(ICRS), Average_G_Magnitude, Average_R_Magnitude, Gaia_G, Gaia_bp, Gaia_rp, Color, Data_Points, #Clipped_G, #Clipped_R, Best_Period(days), Best_Period(minutes), Highest_Lomb-Scargle_Power, (HighestPower-Median)/std" + "\n")

            with open(output_file, "a+") as myfile: 
                myfile.write(' '.join(str(e) for e in output) + "\n")

    # If Output.txt DOES NOT exist
    else:

        # Load in the data on the light curve file
        data = np.loadtxt(files[i])

        # Store the outputs of the LombScargle_MultibandFast function
        output = LombScargle_MultibandFast(files[i], data)

        # If the period is between 60 and 120 minutes with a significance > 8.5
        if((output[15] >= 60) & (output[15] <= 120) & (output[17] >= 8.5)): 

            if not path.exists(output_short_likely_file):
                with open(output_short_likely_file,'a+') as myfile:
                    myfile.write("File_Name, RA(degree), DEC(degree), RA(ICRS), DEC(ICRS), Average_G_Magnitude, Average_R_Magnitude, Gaia_G, Gaia_bp, Gaia_rp, Color, Data_Points, #Clipped_G, #Clipped_R, Best_Period(days), Best_Period(minutes), Highest_Lomb-Scargle_Power, (HighestPower-Median)/std" + "\n")

            with open(output_short_likely_file, "a+") as myfile: 
                myfile.write(' '.join(str(e) for e in output) + "\n")      

        # If the period is between 60 and 120 minutes with a significance < 8.5
            if((output[15] >= 60) & (output[15] <= 120) & (output[17] < 8.5)): 

                if not path.exists(output_short_possible_file):
                    with open(output_short_possible_file,'a+') as myfile:
                        myfile.write("File_Name, RA(degree), DEC(degree), RA(ICRS), DEC(ICRS), Average_G_Magnitude, Average_R_Magnitude, Gaia_G, Gaia_bp, Gaia_rp, Color, Data_Points, #Clipped_G, #Clipped_R, Best_Period(days), Best_Period(minutes), Highest_Lomb-Scargle_Power, (HighestPower-Median)/std" + "\n")

                with open(output_short_possible_file, "a+") as myfile: 
                    myfile.write(' '.join(str(e) for e in output) + "\n")  
                
        if not path.exists(output_file):
            with open(output_file,'a+') as myfile:
                myfile.write("File_Name, RA(degree), DEC(degree), RA(ICRS), DEC(ICRS), Average_G_Magnitude, Average_R_Magnitude, Gaia_G, Gaia_bp, Gaia_rp, Color, Data_Points, #Clipped_G, #Clipped_R, Best_Period(days), Best_Period(minutes), Highest_Lomb-Scargle_Power, (HighestPower-Median)/std" + "\n")

        with open(output_file, "a+") as myfile: 
            myfile.write(' '.join(str(e) for e in output) + "\n")


# In[ ]:





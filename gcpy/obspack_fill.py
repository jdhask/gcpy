"""
Created on Sun Feb 14 12:05:25 2021.

@author: Jessica D.Haskins
"""
import pandas as pd
import numpy as np
import xarray as xr
import datetime


def make_ObsPack_Input_netcdfs(sitename: str, lat: int, lon: int, alt: int,
                               datestart: str, dateend: str, samplefreq: int,
                               sample_stragety: int, outpath: str):
    """Create ObsPack input files for GEOS Chem v13.0.0. for stationary site.
    
    # =========================================================================
    #                             INPUTS
    # =========================================================================
    # sitename = string with sitename used in obspack ID, (e.g.'SOAS-Tower')
    # lat = integer with latitude of obs site in degrees north (e.g. 35.45)
    # lon = integer with longitude of obs site in degrees east
    # alt = integer with elevation plus sample intake height in meters above
    #       sea level
    # datestart = string format YYYYMMDD HH:MM:SS' indicating when you need
    #              obsPack netcdf  file to begin
    # dateend = string format YYYYMMDD HH:MM:SS' indicating when you need
    #              obsPack netcdf  file to stop
    # freq = integer for # of seconds you'd like to step betweeen datestart,
    #         dateend (e.g. 3600 for hourly steps)
    # Sample Stragety = Integer specifiing how the model will average to
    #                   the timebase given (calc'd by  datestart,dateend,
    #                   and freq). Only Valid option are:
    #                             1  =   4-hour avg;
    #                             2  =   1-hour avg;
    #                             3  =   90-min avg;
    #                             4  =   instantaneous
    #
    # outfile_path  = string containing absolute path where the netcdf files
    #                 will be written.
    #
    # =========================================================================
    #                             OUTPUTS
    # =========================================================================
    #
    #  out= Returns list of netcdf files written to outpath for each day in
    #       the time period that  can be used as GEOS Chem inputs.
    #
    # =========================================================================
    #                              Example
    # =========================================================================
    #
    # SOAS Centerville Ground collection site was at lat=32.903281,
    # lon=-87.249942, alt=125. Let's sample the model there ever hour
    # (outputting the hourly average).This code snippiet will make ALL the
    # .nc files we need for ObsPack input for the length of our run (6/1/2013-
    # 7/15/2013) and save them to my desktop.
    #
    # import obspack_fill as obs # place function in same path as call.
    # import xarray
    #
    # filename = obs.make_ObsPack_Input_netcdfs('SOAS-Ground', lat=32.903281,
    #                   lon=-87.249942, alt=125,datestart='20130601 00:00:00',
    #                   dateend='20130716 00:00:00', samplefreq=3600,
    #                   sample_stragety=2, outpath='C:/Users/jhask/Desktop/')
    #
    #  # Open the first file with xarray and print info about it.
    #  dat = xr.open_dataset(filename[0])
    #  print(dat)
    #
    """
    # Figure out # of days between start & Enddate (e.g. # of NC files we need)
    start = datetime.datetime.strptime(datestart, '%Y%m%d %H:%M:%S')
    endd = datetime.datetime.strptime(dateend, '%Y%m%d %H:%M:%S')
    delta = endd - start

    if (delta.seconds != 0):
        days = delta.days + 1  # add a day if there's some leftover time
    else:
        days = delta.days  # otherwise stick with the round number.
    starts = pd.date_range(datestart, dateend, freq='1D').to_series()
    starts2Use = starts[:days]  # only use the # of days we decided on.
    ends2Use = starts2Use + \
        datetime.timedelta(days=1) - datetime.timedelta(seconds=1)
    # don't do the whole day if the stop time is shorter.
    if ends2Use[-1] > endd:
        ends2Use[-1] = endd
    
    # ===================================================================
    all_files = list()  # empty list to contain netcdf filenames generated.
    
    # Loop over # of days we need a file for and make a netcdf!
    for t in range(0, len(starts2Use)):
        # Create the time components variable.
        dfi = pd.DataFrame()  # make empty data frame.
        dts = pd.date_range(  # Get datetimes in our day at sample freq
            starts2Use[t], ends2Use[t], freq=str(samplefreq) + 's')
        
        # Fill columns of dataframe with datetime objs
        dfi['datetime'] = dts
        
        # Strip YMD HMS out of the datetime as integers.
        dfi['year'] = dfi['datetime'].dt.strftime("%Y").astype(int)
        dfi['mon'] = dfi['datetime'].dt.strftime("%m").astype(int)
        dfi['day'] = dfi['datetime'].dt.strftime("%d").astype(int)
        dfi['hr'] = dfi['datetime'].dt.strftime("%H").astype(int)
        dfi['min'] = dfi['datetime'].dt.strftime("%M").astype(int)
        dfi['s'] = dfi['datetime'].dt.strftime("%S").astype(int)
        dfi = dfi.drop(columns='datetime')  # drop datetime column
        
        time_components = dfi.to_numpy()  # convert to a numpy array b4 xarray
        
        # =====================================================================
        
        # Create unique obspack array of ID strings:
        n = np.arange(0, len(dts)).astype(str)  # Create a uniq obs #
        # Create a prefix containing the sitename, start/stop dates
        prefix = sitename + '_from_' + \
            datestart.split(' ')[0] + '_to_' + dateend.split(' ')[0] + '_n'
            
        # Old option to make string array for IDS (not char array?)
        # ids = np.array([(prefix + stri) for stri in n]).astype('|S200')
        
        #  If char array required (we currently think it is.)
        ids = np.chararray((len(dts), 200))
        count = 0
        for stri in n:  # Loop over # of obs and join prefix to obs #, then pad
            # the sting with underscores to make sure we make 200 chars
            indv_row = list((prefix + stri).ljust(200, "_"))  # turn into chars
            ids[count, :] = indv_row  # save IDs in array we pass to xarray.
            count = count + 1
        
        # ====================================================================
        # =========    Make Big XArray with everything required.      ========
        # ====================================================================
        
        # Format of vars & attributes comes from GC example shown here:
        #   http://wiki.seas.harvard.edu/geos-chem/index.php/ObsPack_diagnostic
        
        ds = xr.Dataset({
            # Duplicate lat as many times as you collect obs points. Station
            # isnt' moving.
            'latitude': xr.DataArray(
                        data=np.full(len(dts), lat, dtype=np.float32),
                        dims=['obs'],
                        attrs={
                            "units": "degrees_north",
                            "_FillValue": -1.0e+34,
                            "long_name": "Sample latitude"
                        }),
            # Duplicate lon as many times as you collect obs points.
            'longitude': xr.DataArray(
                data=np.full(len(dts), lon, dtype=np.float32),
                dims=['obs'],
                attrs={
                    "units": "degrees_east",
                    "_FillValue": -1.0e+34,
                    "long_name": "Sample longitude"
                }),
            # Duplicate alt as many times as you collect obs points.
            'altitude': xr.DataArray(
                data=np.full(len(dts), lon, dtype=np.float32),
                dims=['obs'],
                attrs={
                    "units": "meters",
                    "_FillValue": -1.0e+34,
                    "long_name": "sample altitude in meters above sea level",
                    "comment": "Altitude is elevation plus sample intake height in meters above sea level."
                }),
            # This is the only thing changing- array of Y M D H M S to sample
            # this lat,on, alt point. Must be UTC.
            'time_components': xr.DataArray(
                data=time_components,
                dims=['obs', 'calendar_components'],
                attrs={
                    "_FillValue": -9,
                    "long_name": " Calendar time components as integers. Times and dates are UTC.",
                    "order": "year, month, day, hour, minute, second",
                    "comment": "Calendar time components as integers.  Times and dates are UTC."
                }),
            # And now also pass our unique IDs (for each sample point).
            # Dimension is obs #, 200 len string.
            'obspack_id': xr.DataArray(
                data=ids,
                dims=['obs', 'string_of_200chars'],
                attrs={
                    "long_name": "Unique ObsPack observation id",
                    "comment": "Unique observation id string that includes obs_id, dataset_id and obspack_num."
                }),
            # Duplicate the int that indicates sampling stragety as many times
            # as you want to sample the model.
            'CT_sampling_strategy': xr.DataArray(
                data=np.full(len(dts), sample_stragety, dtype=int),
                dims=['obs'],
                attrs={
                    "_FillValue": -9,
                    "long_name": "Unique ObsPack observation id",
                    "values": "How to sample model. 1=4-hour avg; 2=1-hour avg; 3=90-min avg; 4=instantaneous"
                })
        })
        # Create a unique filename for this specific date's obspack input.
        filename = 'obspack_' + sitename + '_freq' + \
            str(samplefreq) + 's.' + str(starts2Use[t]).split(' ')[0] + '.nc'
        
        # Convert our Xarray Data set to a netcdf file w/ this name at outpath
        ds.to_netcdf(outpath + filename)
        
        # Append name of this file to our list of file names that we return
        # to the user.
        all_files.append(outpath + filename)
    
    return all_files

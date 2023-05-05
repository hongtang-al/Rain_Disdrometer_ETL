# Rain_Disdrometer_ETL
Notebooks to pull heavy disdrometer data from database from multiple devices for one year.

#### the key learnings are:
* understanding the query sequence, filter out, shrink heavy tables as much as you can using where and having clause
* understanding database indexing: in our case it is much faster query using time (our database index) than devices
select *
from device_data.dsd_raw
where time >start and time <end
* the query below is much slower (10-100 times slower if pulling on more than 1000 devices)
select *
from device_data.dsd_raw
where device in (device1, device2...)
* using sqlchemy to read and cache large data table into pandas for large table joining etc.
* finally, when working on CTE sequences, working on smaller tables such as CALVAL reference data tables first, it is much lighter overhead

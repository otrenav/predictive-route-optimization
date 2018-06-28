
import tensorflow as tf
import datetime as dt
import pandas as pd
import os


def load_data():
    col_names = [
        'id', 'event_timestamp', 'course_over_ground', 'machine_id',
        'vehicle_weight_type', 'speed_gps_kph', 'latitude', 'longitude']

    data = pd.DataFrame(columns=col_names)
    files = os.listdir('./machine-data')

    for f in files:
        d = pd.read_csv('./machine-data/' + f, sep=';')
        d.loc[d.course_over_ground == -1, 'course_over_ground'] = None
        d['north_proportion'] = north_proportion(d.course_over_ground)
        data = data.append(d)

    # data.to_csv("./machine-data-processed/raw.csv", index=False)
    # data = pd.read_csv('./machine-data-processed/raw.csv')
    return(data)


def north_proportion(vehicle_course_over_ground):
    north_count = sum(
        (vehicle_course_over_ground < 90) |
        (vehicle_course_over_ground > 269)
    )
    return(float(north_count) / len(vehicle_course_over_ground))


def clean(data):
    data = data.drop(columns=['id'])
    data = split_dates_and_times(data, 'event_timestamp')
    data.speed_gps_kph = pd.to_numeric(data.speed_gps_kph)
    data = week_days(data)
    return(data)


def split_dates_and_times(data, column):
    data['year'] = pd.to_numeric(year(data[column]))
    data['month'] = pd.to_numeric(month(data[column]))
    data['day'] = pd.to_numeric(day(data[column]))
    data['hour'] = pd.to_numeric(hour(data[column]))
    data['minute'] = pd.to_numeric(minute(data[column]))
    data = data.drop(columns=[column])
    return(data)


def year(strings):
    return split(strings, 0, '-', 0)


def month(strings):
    return split(strings, 0, '-', 1)


def day(strings):
    return split(strings, 0, '-', 2)


def hour(strings):
    return split(strings, 1, ':', 0)


def minute(strings):
    return split(strings, 1, ':', 1)


def split(strings, first, separator, second):
    return [s.split(' ')[first].split(separator)[second] for s in strings]


def week_days(data):
    data.loc[:, 'weekday'] = data.apply(get_row_weekday, axis=1)
    return(data)


WEEKDAYS = {
    0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday',
    4: 'Friday', 5: 'Saturday', 6: 'Sunday'
}


def get_row_weekday(row):
    date = dt.datetime(row.year, row.month, row.day)
    return WEEKDAYS[date.weekday()]


WEATHER = pd.read_csv('./weather.csv')
WEATHER = split_dates_and_times(WEATHER, 'dt_iso')


def join_with_weather(data):
    for index, row in WEATHER.iterrows():
        selection = (data.day == row.day) & (data.hour == row.hour)
        data.loc[selection, 'weather_description'] = row.weather_description
        data.loc[selection, 'weather_main'] = row.weather_main
        data.loc[selection, 'temperature'] = row.temp
    data.temperature = data.temperature - 273.15
    # data = pd.read_csv("./machine-data-processed/clean.csv")
    # data.to_csv("./machine-data-processed/clean.csv", index=False)
    return(data)


def input_fn(train_X, train_Y, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((dict(train_X), train_Y))
    dataset = dataset.batch(batch_size)
    return dataset

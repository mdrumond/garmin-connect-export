"""
Tests for gcexport.py; Call them with this command line:

py.test gcexport_test.py
"""

from gcexport import *
from StringIO import StringIO

def test_pace_or_speed_raw_cycling():
    # 10 m/s is 36 km/h
    assert pace_or_speed_raw(2, 4, 10.0) == 36.0

def test_pace_or_speed_raw_running():
    # 3.33 m/s is 12 km/h is 5 min/km
    assert pace_or_speed_raw(1, 4, 10.0/3) == 5.0

def test_trunc6_more():
    assert trunc6(0.123456789) == '0.123456'

def test_trunc6_less():
    assert trunc6(0.123) == '0.123000'

def test_offset_date_time():
    assert offset_date_time("2018-03-08 12:23:22", "2018-03-08 11:23:22") == datetime(2018, 3, 8, 12, 23, 22, 0, FixedOffset(60, "LCL"))
    assert offset_date_time("2018-03-08 12:23:22", "2018-03-08 12:23:22") == datetime(2018, 3, 8, 12, 23, 22, 0, FixedOffset(0, "LCL"))

def test_csv_write_record():
    with open('json/activitylist-service.json') as json_data_1:
        activities = json.load(json_data_1)
    with open('json/activity_2541953812.json') as json_data_2:
        details = json.load(json_data_2)
    with open('json/device_99280678.json') as json_data_3:
        device = json.load(json_data_3)
    with open('json/activity_types.properties', 'r') as prop_1:
        activity_type_props = prop_1.read().replace('\n', '')
    activity_type_name = load_properties(activity_type_props)
    with open('json/event_types.properties', 'r') as prop_2:
        event_type_props = prop_2.read().replace('\n', '')
    event_type_name = load_properties(event_type_props)

    parent_type_id = 4
    type_id = 4
    start_latitude = 46.7
    start_longitude = 7.1
    end_latitude = 46.8
    end_longitude = 7.2
    start_time_with_offset = offset_date_time("2018-03-08 12:23:22", "2018-03-08 11:23:22")
    end_time_with_offset = offset_date_time("2018-03-08 12:23:22", "2018-03-08 12:23:22")
    duration = 42.43

    csv_file = StringIO()
    csv_write_record(csv_file, activities[0], details, type_id, parent_type_id, activity_type_name, event_type_name, device,
                     start_time_with_offset, end_time_with_offset, duration, start_latitude, start_longitude,
                     end_latitude, end_longitude)
    assert csv_file.getvalue()[:10] == '"Reckingen'
import sys
from datetime import timedelta, datetime


def progress(count, total, status=''):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', status))
    sys.stdout.flush()

def daterange(start_date, end_date):
    try:
        start_date = datetime.strptime(start_date, "%Y-%m-%d")
        end_date = datetime.strptime(end_date, "%Y-%m-%d")
    except:
        pass

    for n in range(int ((end_date - start_date).days)):
        yield start_date + timedelta(days=n)

def time_to_complete(completed, time_, tasks, total_tasks):
    taken = ((time()-time_)/60)
    time_per_task = taken/tasks
    remaining_tasks = total_tasks-completed
    return remaining_tasks*time_per_task

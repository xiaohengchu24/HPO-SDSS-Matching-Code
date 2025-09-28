import os
import glob
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from astropy.io import fits
from astropy.wcs import WCS
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation, AltAz, ICRS
from astropy import units as u, __version__ as astropy_version
from skyfield.api import load, EarthSatellite, Topos, utc
from concurrent.futures import ThreadPoolExecutor, as_completed
import bz2
import warnings
import re
import argparse
import sys
import hashlib
from astropy.io.fits.verify import VerifyWarning
from functools import lru_cache
import sqlite3
import pickle
import threading
import time

# 忽略 FITS 和 Astropy 警告
warnings.filterwarnings('ignore', category=VerifyWarning)
warnings.filterwarnings('ignore', category=Warning, module='astropy')

# 设置日志
logging.basicConfig(
    filename='streak_matching.log',
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger()

# 记录环境版本
logger.info(f"Python version: {sys.version}")
logger.info(f"Astropy version: {astropy_version}")
try:
    from skyfield import __version__ as skyfield_version
except ImportError:
    skyfield_version = "unknown"


# 参数配置
CONFIG = {
    'fits_dir': r'G:\aaa1111\TOTAL',
    'json_dir': r'G:\aaa1111\visualizations',
    'tle_dir': r'E:\20250803\nnfix',
    'output_csv': r'E:\streak_matching_results_tf28_xxx.csv',
    'fallback_output_csv': r'G:\sdss_19\streak_matching_results_tf28_xxx.csv',
    'angle_threshold': 5.0,  # 方向角度误差阈值（度）
    'fov_radius': 5.0,  # 视场半径（度）
    'tle_time_window': 5,  # TLE 时间匹配范围（天）
    'max_candidates': 2,  # 最佳候选卫星数量
    'max_threads': 10,  # 最大线程数
    'image_size': (2048, 1489),  # 默认图像尺寸
    'time_window_minutes': 2,  # 卫星轨迹预测时间范围（分钟）
    'trajectory_cache_size': 210,  # 轨迹缓存大小（减少以节省内存）
    'trajectory_samples': 100,  # 采样点
}

# 全局缓存
TLE_CACHE = {}
TRAJECTORY_CACHE = {}

TRAJECTORY_CACHE_HITS = 0
TRAJECTORY_CACHE_MISSES = 0
SAT_DATA_CACHE = {}
SAT_DATA_CACHE_HITS = 0
SAT_DATA_CACHE_MISSES = 0

# SQLite for trajectory cache overflow
TRAJECTORY_DB = 'trajectory_cache.db'
with sqlite3.connect(TRAJECTORY_DB) as conn:
    conn.execute('''CREATE TABLE IF NOT EXISTS trajectory_cache (hash_key TEXT PRIMARY KEY, data BLOB)''')

# 新增 TLE 缓存数据库（已存在 tle_cache.db，但增强增量）
TLE_CACHE_DB = 'tle_cache.db'
with sqlite3.connect(TLE_CACHE_DB) as conn:
    conn.execute('''CREATE TABLE IF NOT EXISTS tle_cache (tle_file TEXT PRIMARY KEY, data BLOB, file_hash TEXT)''')

# 新增 FITS 缓存数据库
FITS_CACHE_DB = 'fits_cache.db'
with sqlite3.connect(FITS_CACHE_DB) as conn:
    conn.execute('''CREATE TABLE IF NOT EXISTS fits_cache 
                    (file_hash TEXT PRIMARY KEY, file_path TEXT, data BLOB)''')

# 全局锁 for 轨迹缓存
TRAJECTORY_CACHE_LOCK = threading.Lock()

# 观测站位置
LOC = EarthLocation(lat=32.780361 * u.deg, lon=-105.820417 * u.deg, height=2788 * u.m)


def parse_args():
    parser = argparse.ArgumentParser(description='Satellite streak matching script')
    parser.add_argument('--angle_threshold', type=float, default=CONFIG['angle_threshold'],
                        help='Angle matching threshold in degrees')
    parser.add_argument('--tle_time_window', type=int, default=CONFIG['tle_time_window'],
                        help='TLE time window in days')
    parser.add_argument('--time_window_minutes', type=int, default=CONFIG['time_window_minutes'],
                        help='Satellite path prediction time window in minutes')
    args = parser.parse_args()
    CONFIG['angle_threshold'] = args.angle_threshold
    CONFIG['tle_time_window'] = args.tle_time_window
    CONFIG['time_window_minutes'] = args.time_window_minutes
    return args


def parse_date(date_str, tai_hms, tai_seconds=None):
    try:
        if re.match(r'\d{4}-\d{2}-\d{2}', date_str):
            reformatted = f"{date_str} {tai_hms}"
            tai_time = Time(reformatted, format='iso', scale='tai')
            utc_time = tai_time.utc
            year = utc_time.datetime.year
            if not (1997 <= year <= 2009):
                logger.warning(f"Parsed year {year} outside 1997-2009, please verify DATE-OBS={date_str}")
            logger.debug(f"Parsed date from YYYY-MM-DD: TAI={tai_time}, UTC={utc_time}")
        elif re.match(r'(\d{2})/(\d{2})/(\d{2})', date_str):
            day, month, year = re.match(r'(\d{2})/(\d{2})/(\d{2})', date_str).groups()
            if int(year) >= 97:
                year = f"19{year}"
            else:
                year = f"20{year}"
            reformatted = f"{year}-{month}-{day} {tai_hms}"
            tai_time = Time(reformatted, format='iso', scale='tai')
            utc_time = tai_time.utc
            logger.debug(f"Parsed date from DD/MM/YY: TAI={tai_time}, UTC={utc_time}")
        elif re.match(r'(\d{2})/(\d{2})/(\d{2})', date_str):
            year, month, day = re.match(r'(\d{2})/(\d{2})/(\d{2})', date_str).groups()
            if int(year) >= 97:
                year = f"19{year}"
            else:
                year = f"20{year}"
            reformatted = f"{year}-{month}-{day} {tai_hms}"
            tai_time = Time(reformatted, format='iso', scale='tai')
            utc_time = tai_time.utc
            logger.debug(f"Parsed date from YY/MM/DD: TAI={tai_time}, UTC={utc_time}")
        else:
            raise ValueError(f"Unsupported DATE-OBS format: {date_str}")
        if tai_seconds and isinstance(tai_seconds, (int, float)) and np.isfinite(tai_seconds):
            days_since_epoch = tai_seconds / 86400.0
            mjd = days_since_epoch
            tai_time_check = Time(mjd, format='mjd', scale='tai')
            year_check = tai_time_check.datetime.year
            if 1997 <= year_check <= 2009:
                if abs((tai_time_check - tai_time).jd) > 1e-5:
                    logger.warning(
                        f"TAI seconds {tai_seconds} (MJD={mjd:.6f}, {tai_time_check}) does not match DATE-OBS/TAIHMS {tai_time}")
            else:
                logger.warning(
                    f"TAI seconds {tai_seconds} parsed to invalid year {year_check}, using DATE-OBS/TAIHMS")
        return utc_time
    except Exception as e:
        logger.error(f"Error parsing date: DATE-OBS={date_str}, TAIHMS={tai_hms}, TAI={tai_seconds}: {str(e)}")
        raise


def apply_atmospheric_refraction(coord, obs_time):
    try:
        altaz = coord.transform_to(AltAz(obstime=obs_time, location=LOC))
        alt = altaz.alt.deg
        az = altaz.az.deg
        if not np.isfinite(alt) or alt <= 0:
            alt = max(10.0, 90.0 - abs(coord.dec.deg))
            logger.warning(f"Invalid altitude {alt} deg, estimating based on dec={coord.dec.deg:.2f}")
        alt_rad = np.deg2rad(alt)
        R = 1.02 / np.tan(alt_rad + 10.3 / (alt + 5.11) * np.pi / 180) / 60
        logger.debug(f"Refraction correction: alt={alt:.2f} deg, R={R:.4f} deg")
        apparent_alt = alt + R
        apparent_altaz = AltAz(alt=apparent_alt * u.deg, az=az * u.deg, obstime=obs_time, location=LOC)
        return apparent_altaz.transform_to(ICRS())
    except Exception as e:
        logger.error(f"Error applying atmospheric refraction: {str(e)}")
        return coord


@lru_cache(maxsize=100)
def parse_fits_header_cached(fits_file):
    # 计算文件 hash 作为键
    file_stat = os.stat(fits_file)
    hash_key = hashlib.md5(f"{fits_file}_{file_stat.st_size}_{file_stat.st_mtime}".encode()).hexdigest()

    # 先查数据库
    with sqlite3.connect(FITS_CACHE_DB) as conn:
        cur = conn.cursor()
        cur.execute("SELECT data FROM fits_cache WHERE file_hash = ?", (hash_key,))
        row = cur.fetchone()
        if row:
            logger.info(f"FITS cache hit from SQLite for {fits_file}")
            return pickle.loads(row[0])

    # 如果 miss，解析 FITS
    try:
        if fits_file.endswith('.bz2'):
            with bz2.open(fits_file, 'rb') as f:
                with fits.open(f) as hdu:
                    header = hdu[0].header
        else:
            with fits.open(fits_file) as hdu:
                header = hdu[0].header
        date_obs = header.get('DATE-OBS', 'N/A')
        tai_hms = header.get('TAIHMS', 'N/A')
        tai_seconds = header.get('TAI', None)
        if date_obs == 'N/A' or tai_hms == 'N/A':
            raise ValueError("Missing DATE-OBS or TAIHMS in header")
        obs_time = parse_date(date_obs, tai_hms, tai_seconds)
        exptime = float(header.get('EXPTIME', 53.907456))
        wcs = WCS(header)
        if not wcs.is_celestial:
            raise ValueError("Invalid WCS: not celestial coordinates")
        crval1 = header.get('CRVAL1', 'N/A')
        crval2 = header.get('CRVAL2', 'N/A')
        ctype1 = header.get('CTYPE1', 'N/A')
        ctype2 = header.get('CTYPE2', 'N/A')
        cd1_1 = header.get('CD1_1', 'N/A')
        cd1_2 = header.get('CD1_2', 'N/A')
        cd2_1 = header.get('CD2_1', 'N/A')
        cd2_2 = header.get('CD2_2', 'N/A')
        naxis1 = header.get('NAXIS1', CONFIG['image_size'][0])
        naxis2 = header.get('NAXIS2', CONFIG['image_size'][1])
        if not (isinstance(crval1, (int, float)) and isinstance(crval2, (int, float)) and np.isfinite(
                crval1) and np.isfinite(crval2) and ctype1.startswith('RA---') and ctype2.startswith('DEC--')):
            raise ValueError(f"Invalid WCS: CRVAL1={crval1}, CRVAL2={crval2}, CTYPE1={ctype1}, CTYPE2={ctype2}")
        if not all(isinstance(x, (int, float)) and np.isfinite(x) for x in [cd1_1, cd1_2, cd2_1, cd2_2] if x != 'N/A'):
            logger.warning(f"Incomplete WCS matrix: CD1_1={cd1_1}, CD1_2={cd1_2}, CD2_1={cd2_1}, CD2_2={cd2_2}")
        ra = crval1
        dec = crval2
        center = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame='icrs')
        ra_header = header.get('RA', 'N/A')
        dec_header = header.get('DEC', 'N/A')
        ra_diff = abs(ra_header - crval1) if isinstance(ra_header, (int, float)) and isinstance(crval1,
                                                                                                (int, float)) else 'N/A'
        dec_diff = abs(dec_header - crval2) if isinstance(dec_header, (int, float)) and isinstance(crval2, (
        int, float)) else 'N/A'
        logger.info(
            f"RA, DEC vs CRVAL1, CRVAL2: RA={ra_header}, DEC={dec_header}, ra_diff={ra_diff:.4f}, dec_diff={dec_diff:.4f}")
        alt = float(header.get('ALT', 57.0))
        center = apply_atmospheric_refraction(center, obs_time)
        logger.info(f"FITS {fits_file}: obs_time={obs_time}, exptime={exptime}, ra={ra}, dec={dec}, alt={alt}, "
                    f"CRVAL1={crval1}, CRVAL2={crval2}, CTYPE1={ctype1}, CTYPE2={ctype2}, "
                    f"CD1_1={cd1_1}, CD1_2={cd1_2}, CD2_1={cd2_1}, CD2_2={cd2_2}, "
                    f"NAXIS1={naxis1}, NAXIS2={naxis2}, TAI={tai_seconds}, DATE-OBS={date_obs}, TAIHMS={tai_hms}, "
                    f"corrected_center=[ra={center.ra.deg:.4f}, dec={center.dec.deg:.4f}]")
        result_dict = {
            'obs_time': obs_time,
            'exptime': exptime,
            'wcs': wcs,
            'center': center,
            'alt': alt,
            'naxis': (naxis1, naxis2)
        }
        # 保存到数据库
        with sqlite3.connect(FITS_CACHE_DB) as conn:
            cur = conn.cursor()
            cur.execute("INSERT OR REPLACE INTO fits_cache (file_hash, file_path, data) VALUES (?, ?, ?)",
                        (hash_key, fits_file, pickle.dumps(result_dict)))
            conn.commit()
        logger.info(f"Saved FITS parse result to SQLite for {fits_file}")
        return result_dict
    except Exception as e:
        logger.error(f"Error parsing FITS header {fits_file}: {str(e)}")
        return None


@lru_cache(maxsize=100)
def load_json_labels_cached(json_file, naxis_tuple):
    naxis = naxis_tuple
    try:
        if not os.path.exists(json_file):
            raise FileNotFoundError(f"JSON file not found: {json_file}")
        with open(json_file, 'r') as f:
            data = json.load(f)
        streaks = []
        for shape in data.get('shapes', []):
            if shape['shape_type'] == 'line':
                points = shape['points']
                if len(points) == 2:
                    if all(isinstance(p, (int, float)) and np.isfinite(p) and p >= 0 for point in points for p in
                           point):
                        if (points[0][0] < naxis[0] and points[0][1] < naxis[1] and points[1][0] < naxis[0] and
                                points[1][1] < naxis[1]):
                            streaks.append({
                                'start': points[0],
                                'end': points[1]
                            })
                        else:
                            logger.warning(
                                f"Pixel coordinates out of bounds in {json_file}: {points}, image_size={naxis}")
                    else:
                        logger.warning(f"Invalid pixel coordinates in {json_file}: {points}")
        if not streaks:
            raise ValueError("No valid line shapes found in JSON")
        logger.info(f"Loaded {len(streaks)} streaks from {json_file}: {streaks}")
        return streaks
    except Exception as e:
        logger.error(f"Error loading JSON {json_file}: {str(e)}")
        return None


def find_tle_files_in_window(tle_files, obs_time):
    try:
        obs_date = obs_time.datetime.replace(tzinfo=None)
        tle_dates = []
        for tle_file in tle_files:
            fname = os.path.basename(tle_file)
            date_patterns = [
                r'tle_(\d{4})-(\d{2})-(\d{2})\.txt',
                r'tle_(\d{4})(\d{2})(\d{2})\.txt',
                r'tle_(\d{4})-(\d{2})-(\d{2})_fallback_\d{4}-\d{2}-\d{2}\.txt'
            ]
            tle_date = None
            for pattern in date_patterns:
                match = re.search(pattern, fname)
                if match:
                    year, month, day = match.groups()
                    tle_date = datetime(int(year), int(month), int(day))
                    break
            if tle_date:
                tle_dates.append((tle_file, tle_date))
        if not tle_dates:
            raise ValueError(f"No valid TLE files found. Available files: {tle_files}")
        time_diffs = [(tle_file, abs((tle_date - obs_date).total_seconds()) / (24 * 3600)) for tle_file, tle_date in
                      tle_dates]
        time_diffs.sort(key=lambda x: x[1])
        selected_files = [td[0] for td in time_diffs if td[1] <= CONFIG['tle_time_window']]
        if not selected_files:
            tle_date_range = [min(t[1] for t in tle_dates), max(t[1] for t in tle_dates)]
            raise ValueError(f"No TLE file within {CONFIG['tle_time_window']} days. "
                             f"Obs: {obs_date}, TLE range: {tle_date_range}")
        logger.info(f"Selected {len(selected_files)} TLE files within window for obs time {obs_date}: {selected_files}")
        return selected_files
    except Exception as e:
        logger.error(f"Error finding TLE files in window for {obs_time}: {str(e)}")
        return []


def load_tle_data(tle_file, obs_time):
    global TLE_CACHE
    # 计算 TLE 文件 hash 以检查是否变化
    file_stat = os.stat(tle_file)
    file_hash = hashlib.md5(open(tle_file, 'rb').read()).hexdigest()  # 基于内容 hash

    # 查数据库
    with sqlite3.connect(TLE_CACHE_DB) as conn:
        cur = conn.cursor()
        cur.execute("SELECT data, file_hash FROM tle_cache WHERE tle_file = ?", (tle_file,))
        row = cur.fetchone()
        if row and row[1] == file_hash:
            tle_list = pickle.loads(row[0])
            ts = load.timescale()
            satellites_raw = []
            for d in tle_list:
                try:
                    sat = EarthSatellite(d['line1'], d['line2'], d['name'], ts)
                    satellites_raw.append(sat)
                except Exception as e:
                    logger.warning(f"Error recreating satellite from cached TLE: {str(e)}")
            if satellites_raw:
                logger.info(f"Loaded {len(satellites_raw)} satellites from SQLite for {tle_file} (hash match)")
                obs_time_naive = obs_time.datetime.replace(tzinfo=None)
                satellites = []
                for sat in satellites_raw:
                    epoch = sat.epoch.utc_datetime().replace(tzinfo=None)
                    time_diff = abs((epoch - obs_time_naive).total_seconds()) / (24 * 3600)
                    satellites.append((sat, time_diff))
                satellites.sort(key=lambda x: x[1])
                TLE_CACHE[tle_file] = satellites
                return satellites

    # 如果 miss 或 hash 不匹配，重新加载并更新
    ts = load.timescale()
    satellites_raw = []
    tle_list = []
    with open(tle_file, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f.readlines() if line.strip() and not line.startswith(('#', '//'))]
    if not lines:
        raise ValueError("TLE file is empty or contains only comments")
    i = 0
    while i < len(lines):
        if lines[i].startswith('1 '):
            line1 = lines[i]
            if i + 1 < len(lines) and lines[i + 1].startswith('2 '):
                line2 = lines[i + 1]
                name = f"SAT_{line1[2:7].strip()}"
                try:
                    if len(line1) != 69 or len(line2) != 69:
                        logger.warning(f"Invalid TLE entry at line {i + 1} in {tle_file}: "
                                       f"length mismatch (line1={len(line1)}, line2={len(line2)})")
                        i += 2
                        continue
                    sat = EarthSatellite(line1, line2, name, ts)
                    satellites_raw.append(sat)
                    tle_list.append({'name': name, 'line1': line1, 'line2': line2})
                except Exception as e:
                    logger.warning(f"Error parsing TLE entry at line {i + 1} in {tle_file}: {str(e)} "
                                   f"name={name[:20]}, line1={line1[:20]}, line2={line2[:20]}")
                i += 2
            else:
                logger.warning(f"Invalid TLE entry at line {i + 1} in {tle_file}: "
                               f"missing line 2 after line 1={lines[i][:20]}")
                i += 1
        else:
            logger.warning(f"Skipping non-TLE line at {i + 1} in {tle_file}: {lines[i][:20]}")
            i += 1
    if not satellites_raw:
        raise ValueError("No valid satellites in TLE file")
    # 保存到数据库，包括新 hash
    with sqlite3.connect(TLE_CACHE_DB) as conn:
        cur = conn.cursor()
        cur.execute("INSERT OR REPLACE INTO tle_cache (tle_file, data, file_hash) VALUES (?, ?, ?)",
                    (tle_file, pickle.dumps(tle_list), file_hash))
        conn.commit()
    logger.info(f"Saved/Updated {len(satellites_raw)} satellites to SQLite for {tle_file} (new hash {file_hash})")
    obs_time_naive = obs_time.datetime.replace(tzinfo=None)
    satellites = []
    for sat in satellites_raw:
        epoch = sat.epoch.utc_datetime().replace(tzinfo=None)
        time_diff = abs((epoch - obs_time_naive).total_seconds()) / (24 * 3600)
        satellites.append((sat, time_diff))
    satellites.sort(key=lambda x: x[1])
    TLE_CACHE[tle_file] = satellites
    logger.info(f"Loaded {len(satellites)} satellites from {tle_file}, best time_diff={satellites[0][1]:.2f} days")
    return satellites


def get_best_satellites(tle_files, obs_time):
    all_satellites = {}
    obs_time_naive = obs_time.datetime.replace(tzinfo=None)
    for tle_file in tle_files:
        satellites_raw = load_tle_data(tle_file, obs_time)  # 已处理增量
        for sat, time_diff in satellites_raw:
            satnum = sat.model.satnum
            if satnum not in all_satellites or time_diff < all_satellites[satnum][1]:
                all_satellites[satnum] = (sat, time_diff)
    best_satellites = list(all_satellites.values())
    best_satellites.sort(key=lambda x: x[1])
    logger.info(f"Selected {len(best_satellites)} unique satellites with closest epochs")
    return best_satellites


def compute_trajectory_hash(satellite_name, obs_time_str, exptime):
    return hashlib.md5(f"{satellite_name}_{obs_time_str}_{exptime}".encode()).hexdigest()


def predict_satellite_path(satellite, obs_time, exptime, observer):
    global TRAJECTORY_CACHE, TRAJECTORY_CACHE_HITS, TRAJECTORY_CACHE_MISSES
    obs_time_str = obs_time.iso
    hash_key = compute_trajectory_hash(satellite.name, obs_time_str, exptime)
    # 先查内存缓存
    with TRAJECTORY_CACHE_LOCK:
        if hash_key in TRAJECTORY_CACHE:
            TRAJECTORY_CACHE_HITS += 1
            logger.debug(f"Trajectory cache hit for {satellite.name}, key={hash_key}")
            return TRAJECTORY_CACHE[hash_key]

    # 再查SQLite with retry
    def query_sqlite():
        for attempt in range(3):  # Retry up to 3 times
            try:
                with sqlite3.connect(TRAJECTORY_DB, timeout=10) as conn:  # Set timeout to 10 seconds
                    cur = conn.cursor()
                    # Enable WAL mode if not already
                    cur.execute("PRAGMA journal_mode=WAL;")
                    cur.execute("SELECT data FROM trajectory_cache WHERE hash_key = ?", (hash_key,))
                    row = cur.fetchone()
                    if row:
                        positions = pickle.loads(row[0])
                        TRAJECTORY_CACHE_HITS += 1
                        logger.debug(f"Trajectory SQLite hit for {satellite.name}, key={hash_key}")
                        return positions
                    return None
            except sqlite3.OperationalError as e:
                if "locked" in str(e):
                    logger.warning(f"SQLite locked on query attempt {attempt + 1}, retrying...")
                    time.sleep(0.5 * (attempt + 1))  # Exponential backoff
                else:
                    raise
        raise sqlite3.OperationalError("SQLite query failed after retries")

    positions = query_sqlite()
    if positions is not None:
        return positions

    TRAJECTORY_CACHE_MISSES += 1
    try:
        ts = load.timescale()
        time_window = CONFIG['time_window_minutes'] * 60
        t_start = ts.from_datetime((obs_time.datetime - timedelta(seconds=time_window)).replace(tzinfo=utc))
        t_end = ts.from_datetime((obs_time.datetime + timedelta(seconds=exptime + time_window)).replace(tzinfo=utc))
        logger.info(
            f"Predicting satellite path for {satellite.name}: t_start={t_start.utc_iso()}, t_end={t_end.utc_iso()}")
        times = ts.linspace(t_start, t_end, CONFIG['trajectory_samples'])
        positions = []
        for t in times:
            logger.debug(f"Computing position for {satellite.name} at time {t.utc_iso()}")
            geocentric = satellite.at(t)
            topocentric = geocentric - observer.at(t)
            ra, dec, _ = topocentric.radec()
            ra_deg = ra.degrees
            dec_deg = dec.degrees
            if not (np.isfinite(ra_deg) and np.isfinite(dec_deg)):
                logger.warning(f"Invalid satellite position for {satellite.name}: ra={ra_deg}, dec={dec_deg}")
                continue
            coord = SkyCoord(ra=ra_deg, dec=dec_deg, unit='deg', frame='icrs')
            logger.debug(f"Satellite {satellite.name} position at {t.utc_iso()}: ra={ra_deg:.4f}, dec={dec_deg:.4f}")
            positions.append(coord)
        if len(positions) < 2:
            raise ValueError("Insufficient valid positions computed")
        # Add to cache with lock
        with TRAJECTORY_CACHE_LOCK:
            TRAJECTORY_CACHE[hash_key] = positions
            if len(TRAJECTORY_CACHE) > CONFIG['trajectory_cache_size']:
                oldest_key = next(iter(TRAJECTORY_CACHE))

                # Save to SQLite with retry
                def insert_sqlite():
                    for attempt in range(3):
                        try:
                            with sqlite3.connect(TRAJECTORY_DB, timeout=10) as conn:
                                cur = conn.cursor()
                                cur.execute("PRAGMA journal_mode=WAL;")
                                cur.execute("INSERT OR REPLACE INTO trajectory_cache (hash_key, data) VALUES (?, ?)",
                                            (oldest_key, pickle.dumps(TRAJECTORY_CACHE[oldest_key])))
                                conn.commit()
                            return
                        except sqlite3.OperationalError as e:
                            if "locked" in str(e):
                                logger.warning(f"SQLite locked on insert attempt {attempt + 1}, retrying...")
                                time.sleep(0.5 * (attempt + 1))
                            else:
                                raise
                    raise sqlite3.OperationalError("SQLite insert failed after retries")

                insert_sqlite()
                del TRAJECTORY_CACHE[oldest_key]
        return positions
    except Exception as e:
        import traceback
        logger.error(f"Error predicting satellite path for {satellite.name}: {str(e)}\n{traceback.format_exc()}")
        return None


def vectorized_separation(center, positions):
    center_ra = np.deg2rad(center.ra.deg)
    center_dec = np.deg2rad(center.dec.deg)
    pos_ra = np.deg2rad([pos.ra.deg for pos in positions])
    pos_dec = np.deg2rad([pos.dec.deg for pos in positions])
    sin_dec1 = np.sin(center_dec)
    cos_dec1 = np.cos(center_dec)
    sin_dec2 = np.sin(pos_dec)
    cos_dec2 = np.cos(pos_dec)
    delta_ra = pos_ra - center_ra
    cos_delta_ra = np.cos(delta_ra)
    cos_sep = sin_dec1 * sin_dec2 + cos_dec1 * cos_dec2 * cos_delta_ra
    cos_sep = np.clip(cos_sep, -1.0, 1.0)
    sep = np.rad2deg(np.arccos(cos_sep))
    return sep


def match_streak_to_satellite(streak_start, streak_end, sat_positions, angle_threshold):
    try:
        if len(sat_positions) < 2:
            return False, float('inf')
        avg_dec = (streak_start.dec.deg + streak_end.dec.deg) / 2
        streak_vec = np.array([
            (streak_end.ra.deg - streak_start.ra.deg) * np.cos(np.deg2rad(avg_dec)),
            streak_end.dec.deg - streak_start.dec.deg
        ])
        pos_array = np.array([[p.ra.deg, p.dec.deg] for p in sat_positions])
        diffs = np.diff(pos_array, axis=0)
        avg_sat_dec = np.mean(pos_array[:, 1])
        diffs[:, 0] *= np.cos(np.deg2rad(avg_sat_dec))
        sat_vec = np.mean(diffs, axis=0)
        norm_streak = np.linalg.norm(streak_vec)
        norm_sat = np.linalg.norm(sat_vec)
        if norm_streak == 0 or norm_sat == 0:
            return False, float('inf')
        cos_theta = np.dot(streak_vec, sat_vec) / (norm_streak * norm_sat)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        angle_diff = np.rad2deg(np.arccos(cos_theta))
        angle_diff = min(angle_diff, 180 - angle_diff)
        matched = angle_diff < angle_threshold
        logger.info(f"Streak matching: matched={matched}, angle_diff={angle_diff:.4f} deg")
        return matched, angle_diff
    except Exception as e:
        logger.error(f"Error matching streak to satellite: {str(e)}")
        return False, float('inf')


def process_group(group):
    if not group:
        return []
    rep_info = group[0]
    rep_file = rep_info['file']
    obs_time = rep_info['obs_time']
    exptime = rep_info['exptime']
    tle_files = glob.glob(os.path.join(CONFIG['tle_dir'], '*.txt'))
    if not tle_files:
        logger.error(f"No TLE files found in {CONFIG['tle_dir']}")
        return [{'fits_file': os.path.basename(rep_file), 'status': 'Failed', 'error': 'No TLE files found'}]
    tle_files_in_window = find_tle_files_in_window(tle_files, obs_time)
    if not tle_files_in_window:
        return [{'fits_file': os.path.basename(rep_file), 'status': 'Failed', 'error': 'No valid TLE files in window'}]
    satellites = get_best_satellites(tle_files_in_window, obs_time)
    if not satellites:
        return [{'fits_file': os.path.basename(rep_file), 'status': 'Failed', 'error': 'No satellites loaded'}]
    observer = Topos(latitude_degrees=32.7794, longitude_degrees=-105.8204, elevation_m=2788)
    fits_info = parse_fits_header_cached(rep_file)
    if not fits_info:
        return [{'fits_file': os.path.basename(rep_file), 'status': 'Failed', 'error': 'FITS header parsing error'}]
    center = fits_info['center']
    cache_key = f"{','.join(sorted(tle_files_in_window))}_{obs_time.iso}_{exptime}"
    global SAT_DATA_CACHE, SAT_DATA_CACHE_HITS, SAT_DATA_CACHE_MISSES
    if cache_key in SAT_DATA_CACHE:
        sat_data = SAT_DATA_CACHE[cache_key]
        SAT_DATA_CACHE_HITS += 1
        logger.debug(f"Sat data cache hit for key={cache_key}")
    else:
        SAT_DATA_CACHE_MISSES += 1
        sat_data = {}
        ts = load.timescale()
        t_mid = ts.from_datetime((obs_time + exptime / 2).datetime.replace(tzinfo=utc))

        t_start1 = ts.from_datetime((obs_time).datetime.replace(tzinfo=utc))
        t_start2 = ts.from_datetime((obs_time + exptime / 4).datetime.replace(tzinfo=utc))

        t_end1 = ts.from_datetime((obs_time + (exptime / 4) * 3).datetime.replace(tzinfo=utc))
        t_end2 = ts.from_datetime((obs_time + exptime).datetime.replace(tzinfo=utc))
        time_points = [t_start1, t_start2, t_mid, t_end1, t_end2]

        for sat_tuple in satellites:
            sat, time_diff = sat_tuple
            try:
                in_fov_rough = False
                for t in time_points:

                    geocentric = sat.at(t)
                    topocentric = geocentric - observer.at(t)
                    ra, dec, _ = topocentric.radec()
                    ra_deg = ra.degrees
                    dec_deg = dec.degrees
                    if not (np.isfinite(ra_deg) and np.isfinite(dec_deg)):
                        continue
                    coord = SkyCoord(ra=ra_deg, dec=dec_deg, unit='deg', frame='icrs')
                    sep = center.separation(coord).deg
                    if sep <= CONFIG['fov_radius'] * 3:
                        in_fov_rough = True
                        break
                if not in_fov_rough:
                    logger.debug(
                        f"Skipped {sat.name} due to rough FOV check: sep={sep:.2f} > {CONFIG['fov_radius'] * 3:.2f}")
                    continue
            except Exception as e:
                logger.warning(f"Rough FOV check failed for {sat.name}: {str(e)}")
                continue
            positions = predict_satellite_path(sat, obs_time, exptime, observer)
            if not positions or len(positions) < 2:
                logger.warning(f"Satellite {sat.name} path prediction failed or insufficient points")
                continue
            min_sep = float('inf')
            in_fov = False
            try:
                sep = vectorized_separation(center, positions)
                min_sep = np.min(sep)
                if min_sep < CONFIG['fov_radius']:
                    in_fov = True
            except Exception as e:
                logger.error(f"Error in FOV check for satellite {sat.name}: {str(e)}")
                continue
            sat_data[sat] = {'positions': positions, 'in_fov': in_fov, 'min_sep': min_sep, 'time_diff': time_diff}
        SAT_DATA_CACHE[cache_key] = sat_data
        logger.info(f"Computed and cached sat_data for group, key={cache_key}, num_sats={len(sat_data)}")
    results = []
    for info in group:
        file_results = process_single_file(info['file'], sat_data)
        if file_results:
            results.extend(file_results)
    return results


def process_single_file(fits_file, sat_data):
    try:
        logger.info(f"Processing {fits_file}")
        fits_info = parse_fits_header_cached(fits_file)
        if not fits_info:
            return [
                {'fits_file': os.path.basename(fits_file), 'status': 'Failed', 'error': 'FITS header parsing error'}]
        obs_time = fits_info['obs_time']
        exptime = fits_info['exptime']
        wcs = fits_info['wcs']
        center = fits_info['center']
        naxis = fits_info['naxis']
        alt = fits_info['alt']
        if not (np.isfinite(center.ra.deg) and np.isfinite(center.dec.deg)):
            raise ValueError(f"Invalid center coordinates: ra={center.ra.deg}, dec={center.dec.deg}")
        fits_basename = os.path.splitext(os.path.basename(fits_file))[0]
        if fits_basename.endswith('.fits'):
            fits_basename = os.path.splitext(fits_basename)[0]
        json_file = os.path.join(CONFIG['json_dir'], f"{fits_basename}.json")
        logger.info(f"Attempting to load JSON file: {json_file}")
        streaks = load_json_labels_cached(json_file, tuple(naxis))
        if not streaks:
            return [{'fits_file': fits_basename, 'status': 'Failed', 'error': 'JSON file error'}]
        results = []
        for idx, streak in enumerate(streaks):
            start_pix = streak['start']
            end_pix = streak['end']
            try:
                if (start_pix[0] < 0 or start_pix[1] < 0 or end_pix[0] < 0 or end_pix[1] < 0 or
                        start_pix[0] >= naxis[0] or start_pix[1] >= naxis[1] or end_pix[0] >= naxis[0] or end_pix[1] >=
                        naxis[1]):
                    raise ValueError(
                        f"Pixel coordinates out of bounds: start_pix={start_pix}, end_pix={end_pix}, image_size={naxis}")
                start_coord = SkyCoord.from_pixel(start_pix[0], start_pix[1], wcs)
                end_coord = SkyCoord.from_pixel(end_pix[0], end_pix[1], wcs)
                start_coord = apply_atmospheric_refraction(start_coord, obs_time)
                end_coord = apply_atmospheric_refraction(end_coord, obs_time)
                ra_start = start_coord.ra.deg
                dec_start = start_coord.dec.deg
                ra_end = end_coord.ra.deg
                dec_end = end_coord.dec.deg
                logger.info(f"Pixel to sky coordinates for streak {idx + 1}: start_pix={start_pix}, end_pix={end_pix}, "
                            f"start_coord=[ra={ra_start:.4f}, dec={dec_start:.4f}], "
                            f"end_coord=[ra={ra_end:.4f}, dec={dec_end:.4f}]")
                if not (np.isfinite(ra_start) and np.isfinite(dec_start) and np.isfinite(ra_end) and np.isfinite(
                        dec_end)):
                    raise ValueError(f"Invalid SkyCoord values: start_coord=[ra={ra_start}, dec={dec_start}], "
                                     f"end_coord=[ra={ra_end}, dec={dec_end}]")
            except Exception as e:
                logger.error(f"Error converting pixel to sky coordinates for streak {idx + 1}: {str(e)}")
                results.append({
                    'fits_file': fits_basename,
                    'streak_id': idx + 1,
                    'status': 'Failed',
                    'error': 'Pixel to sky conversion error'
                })
                continue
            best_matched = False
            best_angle_diff = float('inf')
            best_satellite = None
            candidates = []
            separations = []
            for sat, data in sat_data.items():
                min_sep = data['min_sep']
                separations.append(min_sep)
                if min_sep > 2 * CONFIG['fov_radius']:
                    logger.debug(
                        f"Early exit for streak {idx + 1}: min_separation={min_sep:.4f} deg exceeds 2 * fov_radius={2 * CONFIG['fov_radius']:.4f} deg for {sat.name}")
                    candidates.append((sat.name, min_sep, float('inf')))
                    continue
                in_fov = data['in_fov']
                if not in_fov:
                    logger.info(
                        f"Satellite {sat.name} not in FOV for streak {idx + 1}, min_separation={min_sep:.4f} deg")
                    candidates.append((sat.name, min_sep, float('inf')))
                    continue
                positions = data['positions']
                matched, angle_diff = match_streak_to_satellite(start_coord, end_coord, positions,
                                                                CONFIG['angle_threshold'])
                candidates.append((sat.name, min_sep, angle_diff))
                if matched and angle_diff < best_angle_diff:
                    best_matched = matched
                    best_angle_diff = angle_diff
                    best_satellite = sat
            candidates.sort(key=lambda x: x[2])
            logger.info(f"Candidate satellites for streak {idx + 1}: {candidates[:CONFIG['max_candidates']]}")
            if separations:
                logger.info(
                    f"Separation stats for streak {idx + 1}: min={min(separations):.4f}, max={max(separations):.4f}, mean={np.mean(separations):.4f}")
            result = {
                'fits_file': fits_basename,
                'streak_id': idx + 1,
                'status': 'Success' if best_matched else 'No match',
                'matched': best_matched,
                'satellite_name': best_satellite.name if best_matched else 'N/A',
                'min_angle_error': best_angle_diff if best_matched else float('inf'),
                'start_ra': ra_start,
                'start_dec': dec_start,
                'end_ra': ra_end,
                'end_dec': dec_end,
                'error': '' if best_matched else 'Satellite not in FOV or no match within threshold',
                'candidates': candidates[:CONFIG['max_candidates']],
                'obs_time': obs_time.datetime.strftime('%Y/%m/%d %H:%M:%S')
            }
            results.append(result)
        return results
    except Exception as e:
        logger.error(f"Error processing {fits_file}: {str(e)}")
        return [{'fits_file': os.path.basename(fits_file), 'status': 'Failed', 'error': str(e)}]


def main():
    global TRAJECTORY_CACHE_HITS, TRAJECTORY_CACHE_MISSES, SAT_DATA_CACHE_HITS, SAT_DATA_CACHE_MISSES
    try:
        logger.info("Starting batch processing")
        fits_files = glob.glob(os.path.join(CONFIG['fits_dir'], '*.fits')) + glob.glob(
            os.path.join(CONFIG['fits_dir'], '*.fits.bz2'))
        if not fits_files:
            logger.error("No FITS files found")
            return
        file_infos = []
        for f in fits_files:
            fits_info = parse_fits_header_cached(f)
            if fits_info:
                file_infos.append({
                    'file': f,
                    'obs_time': fits_info['obs_time'],
                    'exptime': fits_info['exptime']
                })
            else:
                logger.warning(f"Skipping invalid FITS: {f}")
        file_infos.sort(key=lambda x: x['obs_time'].jd)
        group_threshold = timedelta(minutes=CONFIG['time_window_minutes'] * 2)
        groups = []
        current_group = []
        for info in file_infos:
            if not current_group:
                current_group.append(info)
            else:
                time_diff = abs(info['obs_time'] - current_group[0]['obs_time'])
                if time_diff < group_threshold:
                    current_group.append(info)
                else:
                    groups.append(current_group)
                    current_group = [info]
        if current_group:
            groups.append(current_group)
        logger.info(f"Grouped {len(file_infos)} files into {len(groups)} groups")
        all_results = []
        batch_size = 50  # 分批大小，可根据内存调整
        temp_csvs = []
        for batch_idx in range(0, len(groups), batch_size):
            batch_groups = groups[batch_idx:batch_idx + batch_size]
            batch_results = []
            with ThreadPoolExecutor(max_workers=CONFIG['max_threads']) as executor:
                future_to_group = {executor.submit(process_group, g): g for g in batch_groups}
                for future in as_completed(future_to_group):
                    results = future.result()
                    if results:
                        batch_results.extend(results)
            # 保存批次结果到临时CSV
            if batch_results:
                df_batch_data = []
                for res in batch_results:
                    row = {
                        'fits_file': res['fits_file'],
                        'streak_id': res.get('streak_id', 'N/A'),
                        'status': res['status'],
                        'matched': res.get('matched', False),
                        'satellite_name': res.get('satellite_name', 'N/A'),
                        'min_angle_error': res.get('min_angle_error', 'N/A'),
                        'start_ra': res.get('start_ra', 'N/A'),
                        'start_dec': res.get('start_dec', 'N/A'),
                        'end_ra': res.get('end_ra', 'N/A'),
                        'end_dec': res.get('end_dec', 'N/A'),
                        'error': res.get('error', ''),
                        'obs_time': res.get('obs_time', 'N/A')
                    }
                    candidates = res.get('candidates', [])
                    for i in range(CONFIG['max_candidates']):
                        if i < len(candidates):
                            cand_name, cand_sep, cand_angle_diff = candidates[i]
                            row[f'candidate_{i + 1}_name'] = cand_name
                            row[f'candidate_{i + 1}_separation'] = cand_sep
                            row[f'candidate_{i + 1}_angle_error'] = cand_angle_diff if cand_angle_diff != float(
                                'inf') else 'N/A'
                        else:
                            row[f'candidate_{i + 1}_name'] = 'N/A'
                            row[f'candidate_{i + 1}_separation'] = 'N/A'
                            row[f'candidate_{i + 1}_angle_error'] = 'N/A'
                    df_batch_data.append(row)
                df_batch = pd.DataFrame(df_batch_data)
                temp_path = f"temp_batch_{batch_idx}.csv"
                df_batch.to_csv(temp_path, index=False)
                temp_csvs.append(temp_path)
            # 清空缓存
            TRAJECTORY_CACHE.clear()
            SAT_DATA_CACHE.clear()
            logger.info(f"Batch {batch_idx} processed, caches cleared")
        # 合并所有临时CSV
        if temp_csvs:
            df_all = pd.concat([pd.read_csv(temp) for temp in temp_csvs])
            output_path = CONFIG['output_csv']
            try:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                with open(output_path, 'w') as f:
                    f.write('')
                df_all.to_csv(output_path, index=False)
                logger.info(f"Results saved to {output_path}")
            except PermissionError as e:
                logger.error(
                    f"Permission denied for {output_path}: {str(e)}. Please check write permissions or run as administrator.")
                output_path = CONFIG['fallback_output_csv']
                try:
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    df_all.to_csv(output_path, index=False)
                    logger.info(f"Results saved to fallback path {output_path}")
                except Exception as e:
                    logger.error(f"Failed to save results to fallback path {output_path}: {str(e)}")
            # 删除临时文件
            for temp in temp_csvs:
                os.remove(temp)
        else:
            logger.warning("No results generated")
        logger.info(
            f"Trajectory cache stats: hits={TRAJECTORY_CACHE_HITS}, misses={TRAJECTORY_CACHE_MISSES}, hit_rate={TRAJECTORY_CACHE_HITS / (TRAJECTORY_CACHE_HITS + TRAJECTORY_CACHE_MISSES + 1e-10):.2%}")
        logger.info(
            f"Sat data cache stats: hits={SAT_DATA_CACHE_HITS}, misses={SAT_DATA_CACHE_MISSES}, hit_rate={SAT_DATA_CACHE_HITS / (SAT_DATA_CACHE_HITS + SAT_DATA_CACHE_MISSES + 1e-10):.2%}")
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")


if __name__ == '__main__':
    parse_args()
    main()
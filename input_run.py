import random
import tensorflow as tf
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from multiprocessing import Pool, cpu_count
import re
import json
from QUIC_parser import process_CH, process_SH
from datetime import datetime, timezone
from collections import Counter




def train_validation_test_split(Data, labels, val_size = 0.1, test_size=0.2):
    assert val_size+test_size < 1 and test_size > 0 and val_size > 0, 'Incorrect Validation or Test size proportion'
    X_tr_val, X_test, y_tr_val, y_test = train_test_split(Data, labels, test_size=test_size)
    X_train, X_val, y_train, y_val  = train_test_split(X_tr_val, y_tr_val, test_size=val_size/(1-test_size))
    del X_tr_val, y_tr_val
    return X_train, X_val, X_test, y_train, y_val, y_test

def vectorizaition(otypes, signature):  
    def wrapper(func):
        return np.vectorize(func, otypes = otypes, signature=signature)
    return wrapper

@vectorizaition(otypes=[np.uint8], signature='(),()->(n)')
def payload_alignment(payload: np.array, req_payload_len: int):
    '''
    truncate or pad payload to fixed size 
    '''
    diff = req_payload_len - len(payload)
    return np.concatenate([payload, np.zeros(diff, dtype = np.uint8)]) if diff > 0 else payload[:req_payload_len]

@vectorizaition(otypes=[object], signature='(),()->()')
def service_label(service, SNI):
    SERVICE_SNI_MASKS = {'Audio-AppleMusic': set(['.*od.*itunes.apple.*', '.*audio.*itunes.apple.*']), 
                         'Audio-Spotify': set(['.*audio.*spotify.*akamai.*', '.*audio.*scdn.*']),
                         'Audio-SoundCloud': set(['.*cf.*hls.*media.*sndcdn.*']),
                         'Audio-YandexMusic': set(['.*storage.*yandex.*']),
                         'Audio-VkMusic': set(['.*vkuseraudio.*']),
                         'Video-Facebook': set(['.*video.*fbcdn.net' ,'scontent.*fbcdn.net']),
                         'Video-Netflix': set(['.*nflxvideo.*']), 
                         'Video-YouTube': set(['r.*-.*googlevideo.*']),
                         'Video-Kinopoisk': set(['.*strm.*yandex.*']),
                         'Video-PrimeVideo': set(['.*row.aiv-cdn.*', 'd\w*.cloudfront.net' , '.*avod.*akamai.*']),
                         'Video-Vimeo': set(['\d*vod-adaptive.*akamai.*']),
                         'Video-VkVideo' : set(['.*vkvd.*']),
                         'LiveVideo-YouTube': set(['.*rtmps.youtube.*', '.*upload.youtube.*']), 
                         'LiveVideo-Facebook': set(['.*rtmp-api.facebook.*']),
                         'Reels-Instagram': set(['.*instagram.*fbcdn.*', 'scontent.*cdninstagram.*']),
                         'Reels-TikTok': set(['.*tiktokcdn.*']),
                         'Reels-VkClips': set(['.*vkvd.*']),
                         'Reels-YouTubeShorts': set(['r.*-.*googlevideo.*'])}
    
    if service.find('Web') == 0:
        return 'Web'

    for mask in SERVICE_SNI_MASKS[service]:
        if re.match(mask, SNI):
            return service
    return 'Web'

@vectorizaition(otypes=[object], signature='()->()')
def ipVector(ip_addr: str):
    ipv = 'ipV6'
    parts = ip_addr.split('.')
    if len(parts) == 4:
        digit = 0
        for part in parts:
            digit+=part.isdigit()
        if digit == 4:
            ipv = 'ipV4'
    
    if ipv == 'ipV4':
        return np.array(list(np.zeros(12, dtype = int))+[int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])])
    
    zero_num = 0
    if ip_addr.find('..') > 0:
        ip6parts = ip_addr.split('..')
        part1len = len(ip6parts[0].split('.'))
        part2len = len(ip6parts[1].split('.')) if ip6parts[1]!= '' else 0           
        zero_num =  (8-part1len - part2len)*2
        
    ipvector = []
    empty_cut = False
    for i, part in enumerate(parts):   
        if part == '':
            if not empty_cut:
                ipvector += list(np.zeros(zero_num, dtype = int)) 
                empty_cut = True
        else:
            ipvector+=[int(part, 16) // 256, int(part, 16) % 256]
    assert len(ipvector) == 16, 'incorrect ipvector length'
    #if len(ipvector) != 16:
    #    print(ip_addr, part1len, part2len, ipvector)
    return  np.array(ipvector)



@vectorizaition(otypes=[object], signature='(),()->()')
def multi_web_service_label(service, SNI):
    SERVICE_SNI_MASKS = {'Audio-AppleMusic': set(['.*od.*itunes.apple.*', '.*audio.*itunes.apple.*']), 
                         'Audio-Spotify': set(['.*audio.*spotify.*akamai.*', '.*audio.*scdn.*']),
                         'Audio-SoundCloud': set(['.*cf.*hls.*media.*sndcdn.*']),
                         'Audio-YandexMusic': set(['s.*storage.*yandex.net']),
                         'Audio-VkMusic': set(['.*vkuseraudio.*']),
                         'Video-Facebook': set(['.*video.*fbcdn.net', 'scontent.*fbcdn.net']),#
                         'Video-Netflix': set(['ip.*nflxvideo.*']), 
                         'Video-YouTube': set(['r.*-.*googlevideo.*']),
                         'Video-Kinopoisk': set(['.*.strm.*yandex.net']),
                         'Video-PrimeVideo': set(['.*row.aiv-cdn.*', '.*avod.*akamai.*']),#'d\w*.cloudfront.net' 
                         'Video-Vimeo': set(['\d*vod-adaptive.*akamai.*']),
                         'Video-VkVideo' : set(['.*vkvd.*']),
                         'LiveVideo-YouTube': set(['.*rtmps.youtube.*', '.*upload.youtube.*']), 
                         'LiveVideo-Facebook': set(['.*rtmp-api.facebook.*']),
                         'Reels-Instagram': set(['.*instagram.*fbcdn.*', 'scontent.*cdninstagram.*']),
                         'Reels-TikTok': set(['.*tiktokcdn.*']),
                         'Reels-VkClips': set(['.*vkvd.*']),
                         'Reels-YouTubeShorts': set(['r.*-.*googlevideo.*'])}
    
    WEB_SNI_MASKS = {'Ads-Criteo': set(['.*criteo.*']),
                     'Ads-SmartAdServer': set(['.*smartadserver.*']),
                     'Ads-Casalemedia': set(['.*casalemedia.*']),
                     'Ads-RubiconProject': set(['.*rubiconproject.*']),
                     'Ads-Adnxs': set(['.*adnxs.*']),
                     'Ads-AdSrvr': set(['.*adsrvr.*']),
                     'Ads-AdManMedia': set(['.*admanmedia.*']),
                     'Ads-PubMatic': set(['.*pubmatic.*']),
                     'Ads-Quantserve': set(['.*quantserve.*']),
                     'Ads-ShareThrough': set(['.*sharethrough.*']),
                     'Ads-Stickyadstv': set(['.*stickyadstv.*']),
                     'Ads-Adfox': set(['.*adfox.*']),
                     'Ads-Google': set(['.*google.*ad.*', '.*doubleclick.*', '.*2mdn.*']),
                     'Web-CDN-Akamai': set(['.*akamai.*']),
                     'Web-CDN-Cloudflare': set(['.*cloudflare.*', '.*rlcdn.*']),
                     'Web-CDN-JsDelivr': set(['.*jsdelivr.*']), 
                     
                     'Web-Apple': set(['.*apple.*', '.*itunes.*', '.*mzstatic.*', '.*icloud.*']),
                     'Web-Spotify': set(['.*spotify.*', '.*scdn.*']), 
                     'Web-SoundCloud':set(['.*soundcloud.*', '.*sndcdn.*']),
                     'Web-Vk':set(['.*vk.*', 'sun.*userapi.*', '.*mycdn.me.*', '.*mail.ru']),
                     'Web-Yandex': set(['.*yandex.*', '.*kinopoisk.*']),
                     'Web-Facebook': set(['.*facebook.*', '.*fbcdn.*', '.*cdn.fbsbx.*', '.*instagram.*']),
                     'Web-Netflix': set(['.*nflx.*', '.*netflix.*']),
                     'Web-Google': set(['.*googlevideo.*', '.*.gvt.*', '.*google.*', '.*youtube.*'
                                    '.*ytimg.*', '.*ggpht.*', '.*gstatic.*' ]),
                     'Web-Amazon': set(['.*amazon.*', '.*cloudfront.*', '.*pv-cdn.*', '.*localytics.*',
                                        '.*primevideo.*', '.*aiv-delivery.*', '.*aiv-cdn.*']),
                     'Web-Mozilla': set(['.*mozilla.*']),
                     'Web-Vimeo': set(['.*vimeo.*']),
                     'Web-OBS': set(['.*obsproject.*']),
                     'Web-TikTok': set(['.*tiktok.*','.*ttwstatic.*', 
                                        '.*byteoversea.*', '.*ibytedtos.*']),
                     'Web-IITP': set(['.*iitp.*']),
                     'Web-Twitter': set(['.*twitter.*']),
                     'Web-ScorecardRsrch': set(['.*scorecardresearch.*']),
                     'Web-Grammarly': set(['.*grammarly.*']),
                     'Web-Linkedin': set(['.*linkedin.*']),
                     'Web-GitHub': set(['.*github.*']), 
                     'Web-Bing': set(['.*bing.*']),
                     'Web-Yahoo': set(['.*yahoo.*']),
                     'Web-Microsoft': set(['.*microsoft.*']),
                     'Web-Bidswitch': set(['.*bidswitch.*']),
                     'Web-Reddit': set(['.*reddit.*']),
                     'Web-Baidu': set(['.*baidu.*'])}
        
    
    
    if service.find('Web') < 0:
        for mask in SERVICE_SNI_MASKS[service]:
            if re.match(mask, SNI):
                return service
        
    for srv in SERVICE_SNI_MASKS:
        for mask in SERVICE_SNI_MASKS[srv]:
            if re.match(mask, SNI):
                return 'TargetOutOfClass'
    
    
    for srv in WEB_SNI_MASKS:
        for mask in WEB_SNI_MASKS[srv]:
            if re.match(mask, SNI):
                return srv
    
    return 'Web'

@vectorizaition(otypes=[object], signature='()->()')
def bytes_to_string(byte: int):
    return chr(byte)

def payload_string_to_int(payload_str, sep = ',') -> np.array:
    return np.array([int(i) for i in payload_str.split(sep)], dtype = np.uint8)

@vectorizaition(otypes=[object], signature='()->()')
def payload_string_to_int_vector(payload_str, sep = ',') -> np.array:
    return payload_string_to_int(payload_str, sep = sep)

def MATEC_data(Data:np.array, payload_size = 784) -> tf.Tensor:
    return tf.convert_to_tensor(payload_alignment(Data, req_payload_len = payload_size)/255)

def BGRUA_data(Data:np.array, payload_size = 900, block_size = 150) -> tf.Tensor:
    assert payload_size % block_size == 0, 'The payload_size must be divisible by the block_size'
    tensor_out =  tf.convert_to_tensor(payload_alignment(Data, req_payload_len = payload_size)/255)
    return tf.reshape(tensor_out, [tensor_out.shape[0], tensor_out.shape[1]*payload_size//block_size, block_size])
    
def ABRF_data(Data:np.array, payload_size = 600) -> np.array:
    Data_abrf = payload_alignment(Data, req_payload_len = payload_size)
    return np.reshape(Data_abrf, (Data_abrf.shape[0], Data_abrf.shape[1]*payload_size))

def hABRF_data(Data_p, Data_f, payload_size = 600) -> np.array:
    return np.concatenate([ABRF_data(Data_p, payload_size = payload_size), Data_f], axis = 1)

def RBRF_data(Data, njobs = 1) -> np.array:
    n_proc = range(cpu_count()+1)[njobs]
    with Pool(processes=n_proc) as pool:
        #result = pool.map(payload_decomposition, Data) if one_vector else pool.map(CH_and_SH_decomposition, Data)
        result = pool.map(CH_and_SH_recomp, Data, chunksize=len(Data) // n_proc)
    return np.array(result, dtype = np.uint8)

def RBRF_data_without_GREASE(Data, njobs = 1) -> np.array:
    n_proc = range(cpu_count()+1)[njobs]
    with Pool(processes=n_proc) as pool:
        #result = pool.map(payload_decomposition, Data) if one_vector else pool.map(CH_and_SH_decomposition, Data)
        result = pool.map(CH_and_SH_recomp_without_GREASE, Data, chunksize=len(Data) // n_proc)
    return np.array(result, dtype = np.uint8)

def JA3_data(Data, njobs = 1) -> np.array:
    n_proc = range(cpu_count()+1)[njobs]
    with Pool(processes=n_proc) as pool:
        #result = pool.map(payload_decomposition, Data) if one_vector else pool.map(CH_and_SH_decomposition, Data)
        result = pool.map(JA3_and_JA3S_fingerprint, Data, chunksize=len(Data) // n_proc)
    return np.array(result, dtype = np.uint8)

def hRBRF_data(Data_p, Data_f, njobs = 1) -> np.array:
    return np.concatenate([RBRF_data(Data_p, njobs = njobs), Data_f], axis = 1)

def InterFlow_hRBRF_data(Data_p, Data_f, Data_bof, njobs = 1) -> np.array:
    return np.concatenate([RBRF_data(Data_p, njobs = njobs), Data_f, Data_bof], axis = 1)

def hypridC45_data(Data_p, Data_f, njobs = 1) -> np.array:
    return np.concatenate([hybridC45_FlowData(Data_f), hypridC45_TLSdata(Data_p, njobs = njobs)], axis = 1)

def hypridC45_TLSdata(Data_p, njobs = 1) -> np.array:
    n_proc = range(cpu_count()+1)[njobs]
    with Pool(processes=n_proc) as pool:
        #result = pool.map(payload_decomposition, Data) if one_vector else pool.map(CH_and_SH_decomposition, Data)
        result = pool.map(hybridC45_TLSfeatures, Data_p, chunksize=len(Data_p) // n_proc)
    return np.array(result, dtype = np.uint8)

def hybridC45_FlowData(Data_f) -> np.array:
    F1 = np.array([Data_f[:,6]]).reshape(Data_f[:,6].shape[0],1)
    F2 = Data_f[:,8:14]
    F3 = Data_f[:,15:20]
    F4 = Data_f[:,24:27]
    F5 = Data_f[:,31:34]
    return np.concatenate([F1,F2,F3,F4,F5], axis = 1)

def EA_data(Data:np.array, njobs = 1, one_vector = False):
    return tf.convert_to_tensor(data_payload_decomposition(Data, njobs = njobs, one_vector = one_vector)/255)

#def data_payload_decomposition(Data, njobs = 1, one_vector = False):
def data_payload_decomposition(Data, njobs = 1) -> np.array:
    n_proc = range(cpu_count()+1)[njobs]
    with Pool(processes=n_proc) as pool:
        #result = pool.map(payload_decomposition, Data) if one_vector else pool.map(CH_and_SH_decomposition, Data)
        result = pool.map(CH_and_SH_decomposition, Data, chunksize=len(Data) // n_proc)
    return np.array(result, dtype = np.uint8)

def one_vector_payload_decomposition(ClientHello_ServerHello: np.array, GREASE = False) -> np.array:
    return [CH_and_SH_decomposition(ClientHello_ServerHello, GREASE = GREASE)]

def data_payload_recomposition(Data, njobs = 1) -> np.array:
    n_proc = range(cpu_count()+1)[njobs]
    with Pool(processes=n_proc) as pool:
        #result = pool.map(payload_decomposition, Data) if one_vector else pool.map(CH_and_SH_decomposition, Data)
        result = pool.map(CH_and_SH_recomp, Data, chunksize=len(Data) // n_proc)
    return np.array(result, dtype = np.uint8)

def data_payload_recomposition_plus_Cert(Data, njobs = 1) -> np.array:
    n_proc = range(cpu_count()+1)[njobs]
    with Pool(processes=n_proc) as pool:
        #result = pool.map(payload_decomposition, Data) if one_vector else pool.map(CH_and_SH_decomposition, Data)
        result = pool.map(recomp_plus_Cert, Data, chunksize=len(Data) // n_proc)
    return np.array(result, dtype = np.uint8)

def recomp_plus_Cert(CH_SH_Cert, req_payload_len = 185):
    return np.concatenate([CH_and_SH_recomp([CH_SH_Cert[0],CH_SH_Cert[1]]), 
                          payload_alignment(CH_SH_Cert, req_payload_len = req_payload_len)[2]])

def JA3_and_JA3S_fingerprint(ClientHello_ServerHello: np.array)->np.array:
    recomposed_CH_SH = CH_and_SH_recomp_without_GREASE(ClientHello_ServerHello)
    return np.array(list(recomposed_CH_SH[6:8]) + list(recomposed_CH_SH[10:80]) + list(recomposed_CH_SH[83:123])+list(recomposed_CH_SH[243:245])+list(recomposed_CH_SH[246:248]) + list(recomposed_CH_SH[250:290]))

def majority_voting(predictions: np.array, ips: np.array) -> np.array:
   
    voting_dict = {}

    # Aggregate predictions for each IP
    for ip, prediction in zip(ips, predictions):
        if ip not in voting_dict:
            voting_dict[ip] = []
        voting_dict[ip].append(prediction)

    # Determine the majority vote for each IP
    majority_votes = {ip: Counter(preds).most_common(1)[0][0] for ip, preds in voting_dict.items()}

    # Assign the majority vote to each prediction based on the IP
    majority_voting_predictions = [majority_votes[ip] for ip in ips]

    return majority_voting_predictions


def majority_voting_with_time(predictions: np.array, ips: np.array, timestamps: np.array) -> np.array:
    # Create a dictionary to store IPs, predictions, and timestamps
    voting_dict = {}
    
    # Result list with majority votes
    majority_voting_predictions = []
    
    # Iterate over all predictions, ips, and timestamps
    for current_index, (prediction, ip, timestamp) in enumerate(zip(predictions, ips, timestamps)):
        # Initialize the IP entry if not present
        if ip not in voting_dict:
            voting_dict[ip] = []
        
        # Filter out the predictions with a timestamp greater or equal to the current one
        previous_votes = [prev_prediction for prev_index, (prev_prediction, prev_timestamp) 
                          in enumerate(voting_dict[ip]) if prev_timestamp < timestamp]
        
        # Add current prediction to the list of previous votes for majority voting
        previous_votes.append(prediction)
        
        # Determine the majority vote considering only previous votes
        vote_count = Counter(previous_votes)
        majority_vote = vote_count.most_common(1)[0][0]
        
        # Add the majority vote to the result list
        majority_voting_predictions.append(majority_vote)
        
        # Append current prediction and timestamp to the voting dictionary
        voting_dict[ip].append((prediction, timestamp))
    
    return majority_voting_predictions

def parameters_position():
    Byte_params = {0: 'CH RecordVersion 1st',
    1: 'CH RecordVersion 2nd', 
    2: 'CH Record Length 1st', 
    3: 'CH Record Length 2nd',
    4: 'CH Message Length 1st', 
    5: 'CH Message Length 2nd', 
    6: 'CH Message Version 1st',
    7: 'CH Message Version 2nd',
    8: 'CH Session ID Length', 
    9: 'CH Cipher Suites Length',
    80: 'CH Extensions Length 1st',
    81: 'CH Extensions Length 2nd',
    82: 'CH Extensions Number',
    123: 'CH SNI Lenght 1st',
    124: 'CH SNI Lenght 2nd',
    125: 'CH ALPN length 1st', 
    126: 'CH ALPN length 2nd', 
    127: 'CH Padding Length 1st', 
    128: 'CH Padding Length 2nd',
    129: 'CH Session Ticket Length 1st',
    130: 'CH Session Ticket Length 2nd',
    131: 'CH PSK Length 1st',
    132: 'CH PSK Length 2nd',  
    133: 'CH Key Share Length 1st',
    134: 'CH Key Share Length 2nd',
    135: 'CH Cookie Length 1st', 
    136: 'CH Cookie Length 2nd', 
    137: 'CH Cached Info Length 1st',
    138: 'CH Cached Info Length 2nd', 
    139: 'CH Trusted CA keys Data 1st', 
    140: 'CH Trusted CA keys Data 2nd', 
    141: 'CH Heartbeat Data 1st', 
    142: 'CH Heartbeat Data 2nd', 
    143: 'CH PSK key exchange modes Data 1st',
    144: 'CH PSK key exchange modes Data 2nd',
    145: 'CH Compress Certificate Data length',
    153: 'CH User Mapping Data Length', 
    157: 'CH EC point formats Data Length', 
    161: 'CH Client Cert Type Data Length', 
    165: 'CH Server Cert Type Data Length',  
    173: 'CH Supported Versions Data Length',  
    185: 'CH Supported Groups Data Length',
    212: 'CH Signature Hash Algorithms Data Length',
    306: 'CH ALPN Data lenght',
    307: 'CH ALPN #1 lenght',
    308: 'CH ALPN #1 1st',
    309: 'CH ALPN #1 2nd',
    237: 'SH RecordVersion 1st',
    238: 'SH RecordVersion 2nd', 
    239: 'SH Record Length 1st', 
    240: 'SH Record Length 2nd',
    241: 'SH Message Length 1st', 
    242: 'SH Message Length 2nd', 
    243: 'SH Message Version 1st',
    244: 'SH Message Version 2nd',
    245: 'SH Session ID Length', 
    246: 'SH Cipher 1st', 
    247: 'SH Cipher 2nd', 
    248: 'SH Extensions Length 1st',
    249: 'SH Extensions Length 2nd',
    290: 'SH Extensions Number',
    294: 'SH PSK Length 1st',
    295: 'SH PSK Length 2nd', 
    296: 'SH Key Share Length 1st',
    297: 'SH Key Share Length 2nd', 
    299: 'SH Supported Version 1st',
    300: 'SH Supported Version 2nd',
    301: 'SH Key Share (supported group) 1st',
    302: 'SH Key Share (supported group) 2nd', 
    310: 'TSPS: UL_num',
    311: 'TSPS: DL_num',
    312: 'TSPS: UL_PS_max',
    313: 'TSPS: UL_PS_min',
    314: 'TSPS: UL_PS_ave',
    315: 'TSPS: UL_PS_std',
    316: 'TSPS: UL_PS_25th',
    317: 'TSPS: UL_PS_50th',
    318: 'TSPS: UL_PS_75th',
    319: 'TSPS: DL_PS_max',
    320: 'TSPS: DL_PS_min',
    321: 'TSPS: DL_PS_ave',
    322: 'TSPS: DL_PS_std',
    323: 'TSPS: DL_PS_25th',
    324: 'TSPS: DL_PS_50th',
    325: 'TSPS: DL_PS_75th',
    326: 'TSPS: UL_TI_max',
    327: 'TSPS: UL_TI_min',
    328: 'TSPS: UL_TI_ave',
    329: 'TSPS: UL_TI_std',
    330: 'TSPS: UL_TI_25th',
    331: 'TSPS: UL_TI_50th',
    332: 'TSPS: UL_TI_75th',
    333: 'TSPS: DL_TI_max',
    334: 'TSPS: DL_TI_min',
    335: 'TSPS: DL_TI_ave',
    336: 'TSPS: DL_TI_std',
    337: 'TSPS: DL_TI_25th',
    338: 'TSPS: DL_TI_50th',
    339: 'TSPS: DL_TI_75th'}

    for i in range(495, 495+20, 2):
        Byte_params[i] = 'CH_CS_len_{}'.format((i-495)/2)
    for i in range(496, 496+20, 2):
        Byte_params[i] = 'SH_CS_{}'.format((i-496)/2)

    for i in range(146,149):
        Byte_params[i] = 'CH Compress Certificate Data {}'.format(i-145)

    for i in range(149,153):
        Byte_params[i] = 'CH Record Size Limit Data {}'.format(i-149)

    for i in range(154,157):
        Byte_params[i] = 'CH User Mapping Data {}'.format(i-153)

    for i in range(158,161):
        Byte_params[i] = 'CH EC point formats Data {}'.format(i-157)

    for i in range(162,165):
        Byte_params[i] = 'CH Client Cert Type Data {}'.format(i-161)

    for i in range(166,169):
        Byte_params[i] = 'CH Client Cert Type Data {}'.format(i-165)

    for i in range(169,173):
        Byte_params[i] = 'CH Ticket Request Data {}'.format(i-168)

    for i in range(174,185):
        Byte_params[i] = 'CH Supported Versions #{0} {1}'.format((i-172)//2,  i%2 + 1)

    for i in range(186,211):
        Byte_params[i] = 'CH Supported Groups #{0} {1}'.format((i-184)//2,  i%2 + 1)

    for i in range(213,237):
        Byte_params[i] = 'CH Signature Hash Algorithms Data #{0} {1}'.format((i-211)//2,  i%2 + 1)


    for i in range(10,80):
        Byte_params[i] = 'CH Cipher Suite #{0} {1}'.format((i-8)//2, i%2 + 1) 

    for i in range(83,123):
        Byte_params[i] = 'CH Ext Type #{0} {1}'.format((i-81)//2, (i+1)%2 + 1) 

    for i in range(250,290):
        Byte_params[i] = 'SH Ext Type #{0} {1}'.format((i-248)//2, i%2 + 1)
    return  Byte_params


def get_feature_importance_w_parameter_names(rf_features_importance: list, print_out = False, filename = None) -> pd.DataFrame:
    Byte_params = parameters_position()
    Imporance = np.array(rf_features_importance)/max(rf_features_importance)
    FI_dict = dict()
    for i, item in enumerate(Imporance):
        if item > 0:
            FI_dict[i] = round(item * 10**3, 2)
            #print(i, item* 10**3)
            
    Sort_dict = {k: v for k, v in sorted(FI_dict.items(), key=lambda x: x[1], reverse=True)}
    output_dict = {'parameter':[], 'Feature-Importance':[]}
    
    for key in Sort_dict.keys():
        if print_out:
            if key in Byte_params:      
                print(Byte_params[key].center(40, " "), str(Sort_dict[key]).center(10, " "))
            else:
                print(str(key).center(30, " "), str(Sort_dict[key]).center(10, " "))
        if key in Byte_params: 
            output_dict['parameter'].append(Byte_params[key])  
            output_dict['Feature-Importance'].append(Sort_dict[key])  
        else:
            output_dict['parameter'].append(str(key))  
            output_dict['Feature-Importance'].append(Sort_dict[key]) 
    
    output_dict = pd.DataFrame(output_dict)
    if filename is not None:
        output_dict.to_csv(filename, index = False)
    return output_dict
                           

def CH_and_SH_recomp(ClientHello_ServerHello: np.array) -> np.array:
    count=0
    payload = np.zeros(310, dtype = np.uint8) #306
    try: 
        ClientHello = ClientHello_ServerHello[0]
        
        #TLS has value 22 in its first byte in handshake
        #if ClientHello[0]!=22:
            #TODO add packet number and QUIC version. Delete this line
        #    ClientHello = np.concatenate([np.array([1,0,0,0,0]),ClientHello])
        
        CH_L_dict = {0: 123, 16: 125, 21: 127, 35: 129, 41: 131, 51: 133, 44: 135, 25:137}
        
        CH_D_dict = {3: {'position': 139, 'len':2},  15: {'position': 141, 'len':2}, 45: {'position': 143, 'len':2},
                    27: {'position': 145, 'len':4},  28: {'position': 149, 'len':4},  6: {'position': 153, 'len':4},
                    11: {'position': 157, 'len':4},  19: {'position': 161, 'len':4}, 20: {'position': 165, 'len':4},
                    58: {'position': 169, 'len':4},  43: {'position': 173, 'len':12}, 16: {'position':306, 'len':4},
                    10: {'position': 185, 'len':26}, 13: {'position': 211, 'len':26}}
        
        
        #RV, RL       #ML, MV       #SID len
        payload[0:4], payload[4:8], payload[8] = ClientHello[1:5], ClientHello[7:11], ClientHello[43]
        #Ciphers Len
        payload[9] = (ClientHello[44 + ClientHello[43]]*256 + ClientHello[45 + ClientHello[43]]) // 2
        L = 46 + ClientHello[43]
        r = payload[11]*2 if payload[11] <= 35 else 70
        #Cipher Suites
        payload[10: 10 + r] =  ClientHello[L:L + r]
        L += payload[9]*2 + ClientHello[L + payload[9]*2]+1
        #Extensions Len
        payload[80: 80+2] = ClientHello[L:L+2]
        L+=2

        type_position = 83 
        #extensions = ClientHello[L:]
        end_of_CH = payload[80]*256 + payload[81] + L
        
        while L < end_of_CH:
            ext_type = ClientHello[L]*256 + ClientHello[L+1]
            ext_len  = ClientHello[L+2]*256 + ClientHello[L+3]
            #ext_data = ClientHello[L+4:L+4+ext_len]
        
            payload[type_position: type_position + 2] = ClientHello[L:L+2]
            type_position+=2 if type_position < 123 else 0
            
            if ext_type in CH_L_dict:
                payload[CH_L_dict[ext_type]:CH_L_dict[ext_type] + 2] = ClientHello[L+2:L+4]
                L+= 4+ext_len
                continue
            elif ext_type in CH_D_dict:
                R = ext_len if ext_len <= CH_D_dict[ext_type]['len'] else CH_D_dict[ext_type]['len']
                payload[CH_D_dict[ext_type]['position']:CH_D_dict[ext_type]['position']+R] = ClientHello[L+4:L+4+R]
                L+= 4+ext_len
                continue
            else:
                L+= 4+ext_len

        #Ext_num
        payload[82] = (type_position-83)//2
            
        ServerHello = ClientHello_ServerHello[1]

        # if ServerHello[0]!=22:
        #     #TODO add packet number and QUIC version
        #     ServerHello = np.concatenate([np.array([2,0,0,0,0]), ServerHello])

        #Record version, Record Len, Handshake Type, Msg len, Msg Ver, SID len
        payload[237:241], payload[241:245], payload[245] = ServerHello[1:5], ServerHello[7:11], ServerHello[43]
        #cipher
        payload[246:248] = ServerHello[44 + ServerHello[43]:46 + ServerHello[43]]
        #ext_len
        payload[248:250] = ServerHello[47 + ServerHello[43]:49 + ServerHello[43]]

        '''
        SH_L_dict = {18: 290, 35: 292, 41: 294, 51: 296}

        SH_D_dict = {43: {'position': 298 , 'len':2},
                    51: {'position': 300, 'len':2},
                    11: {'position': 302, 'len':4}}
        '''
        SH_L_dict = {41: 294, 51: 296}

        SH_D_dict = {43: {'position': 299 , 'len':2},
                    51: {'position': 301, 'len':2}}
        
        
        L = 49 + ServerHello[43]
        end_of_SH = payload[248]*256 + payload[249] + L
        
        type_position = 250 

        while L < end_of_SH:
            ext_type = ServerHello[L]*256 + ServerHello[L+1]
            ext_len  = ServerHello[L+2]*256 + ServerHello[L+3]
            #ext_data = ServerHello[L+4:L+4+ext_len]
        
            payload[type_position: type_position + 2] = ServerHello[L:L+2]
            type_position+=2 if type_position < 290 else 0
            if ext_type in SH_L_dict:
                payload[SH_L_dict[ext_type]:SH_L_dict[ext_type] + 2] = ServerHello[L+2:L+4]
                L+= 4+ext_len
                continue
            elif ext_type in SH_D_dict:
                R = ext_len if ext_len <= SH_D_dict[ext_type]['len'] else SH_D_dict[ext_type]['len']
                payload[SH_D_dict[ext_type]['position']:SH_D_dict[ext_type]['position']+R] = ServerHello[L+4:L+4+R]
                L+= 4+ext_len
                continue
            else:
                L+= 4+ext_len

        #Ext_num
        payload[290] = (type_position-251)//2

    except: 
        count+=1
        print(f'Cannot recomp flow')
    return payload


def CH_and_SH_recomp_without_GREASE(ClientHello_ServerHello: np.array) -> np.array:
    count=0
    payload = np.zeros(310, dtype = np.uint8) #306
    try: 
        ClientHello = ClientHello_ServerHello[0]

        with open('parameters_names/CipherSuites.json', "r") as fp:
            CipherSuites_names = json.load(fp) 

        with open('parameters_names/ext_names.json', "r") as fp:
            ext_names = json.load(fp) 
        with open('parameters_names/supported_groups_names.json', "r") as fp:
            sg_names = json.load(fp) 
        
        #TLS has value 22 in its first byte in handshake
        #if ClientHello[0]!=22:
            #TODO add packet number and QUIC version. Delete this line
        #    ClientHello = np.concatenate([np.array([1,0,0,0,0]),ClientHello])
        
        CH_L_dict = {0: 123, 16: 125, 21: 127, 35: 129, 41: 131, 51: 133, 44: 135, 25:137}
        
        CH_D_dict = {3: {'position': 139, 'len':2},  15: {'position': 141, 'len':2}, 45: {'position': 143, 'len':2},
                    27: {'position': 145, 'len':4},  28: {'position': 149, 'len':4},  6: {'position': 153, 'len':4},
                    11: {'position': 157, 'len':4},  19: {'position': 161, 'len':4}, 20: {'position': 165, 'len':4},
                    58: {'position': 169, 'len':4},  43: {'position': 173, 'len':12}, 16: {'position':306, 'len':4},
                    10: {'position': 185, 'len':26}, 13: {'position': 211, 'len':26}}
        
        
        #RV, RL       #ML, MV       #SID len
        payload[0:4], payload[4:8], payload[8] = ClientHello[1:5], ClientHello[7:11], ClientHello[43]
        #if payload[2]*256+payload[3] != payload[4]*256+payload[5]
        #Ciphers Len
        payload[9] = (ClientHello[44 + ClientHello[43]]*256 + ClientHello[45 + ClientHello[43]]) // 2
        L = 46 + ClientHello[43]
        r = payload[11]*2 if payload[11] <= 35 else 70
        #Cipher Suites
        payload[10: 10 + r] =  ClientHello[L:L + r]
        for i in range(10,10+r,2):
            CipherSuite =  payload[i]*256 + payload[i+1]
            if not CipherSuite in CipherSuites_names:
                payload[i], payload[i+1] = 0,0

        L += payload[9]*2 + ClientHello[L + payload[9]*2]+1
        #Extensions Len
        payload[80: 80+2] = ClientHello[L:L+2]
        L+=2

        type_position = 83 
        #extensions = ClientHello[L:]
        end_of_CH = payload[80]*256 + payload[81] + L
        
        while L < end_of_CH:
            ext_type = ClientHello[L]*256 + ClientHello[L+1]
            ext_len  = ClientHello[L+2]*256 + ClientHello[L+3]
            #ext_data = ClientHello[L+4:L+4+ext_len]
            if (ext_type <=59) or (str(ext_type) in ext_names):
                payload[type_position: type_position + 2] = ClientHello[L:L+2]
            type_position+=2 if type_position < 123 else 0
            
            if ext_type in CH_L_dict:
                payload[CH_L_dict[ext_type]:CH_L_dict[ext_type] + 2] = ClientHello[L+2:L+4]
                L+= 4+ext_len
                continue
            elif ext_type in CH_D_dict:
                R = ext_len if ext_len <= CH_D_dict[ext_type]['len'] else CH_D_dict[ext_type]['len']
                if ext_type == 43:
                    for i in range(L+5,L+4+R,2):
                        TLS_v = ClientHello[i]*256 + ClientHello[i+1]
                        if TLS_v < 3*256 or TLS_v > 3*256+4:
                            ClientHello[i], ClientHello[i+1] = 0,0
                payload[CH_D_dict[ext_type]['position']:CH_D_dict[ext_type]['position']+R] = ClientHello[L+4:L+4+R]
                L+= 4+ext_len
                continue
            else:
                L+= 4+ext_len

        #Ext_num
        payload[82] = (type_position-83)//2
            
        ServerHello = ClientHello_ServerHello[1]

        # if ServerHello[0]!=22:
        #     #TODO add packet number and QUIC version
        #     ServerHello = np.concatenate([np.array([2,0,0,0,0]), ServerHello])

        #Record version, Record Len, Handshake Type, Msg len, Msg Ver, SID len
        payload[237:241], payload[241:245], payload[245] = ServerHello[1:5], ServerHello[7:11], ServerHello[43]
        #cipher
        payload[246:248] = ServerHello[44 + ServerHello[43]:46 + ServerHello[43]]
        #ext_len
        payload[248:250] = ServerHello[47 + ServerHello[43]:49 + ServerHello[43]]

        '''
        SH_L_dict = {18: 290, 35: 292, 41: 294, 51: 296}

        SH_D_dict = {43: {'position': 298 , 'len':2},
                    51: {'position': 300, 'len':2},
                    11: {'position': 302, 'len':4}}
        '''
        SH_L_dict = {41: 294, 51: 296}

        SH_D_dict = {43: {'position': 299 , 'len':2},
                    51: {'position': 301, 'len':2}}
        
        
        L = 49 + ServerHello[43]
        end_of_SH = payload[248]*256 + payload[249] + L
        
        type_position = 250 

        while L < end_of_SH:
            ext_type = ServerHello[L]*256 + ServerHello[L+1]
            ext_len  = ServerHello[L+2]*256 + ServerHello[L+3]
            #ext_data = ServerHello[L+4:L+4+ext_len]
        
            payload[type_position: type_position + 2] = ServerHello[L:L+2]
            type_position+=2 if type_position < 290 else 0
            
            if ext_type in SH_L_dict:
                payload[SH_L_dict[ext_type]:SH_L_dict[ext_type] + 2] = ServerHello[L+2:L+4]
                L+= 4+ext_len
                continue
            elif ext_type in SH_D_dict:
                R = ext_len if ext_len <= SH_D_dict[ext_type]['len'] else SH_D_dict[ext_type]['len']
                payload[SH_D_dict[ext_type]['position']:SH_D_dict[ext_type]['position']+R] = ServerHello[L+4:L+4+R]
                L+= 4+ext_len
                continue
            else:
                L+= 4+ext_len

        #Ext_num
        payload[290] = (type_position-251)//2

    except: 
        count+=1
        print(f'Cannot recomp flow')
    return payload

def CH_and_SH_decomposition(ClientHello_ServerHello: np.array, GREASE = False) -> np.array:

    def CH_short_decomposition(ClientHello: np.array, GREASE = True) -> np.array:
        payload = np.zeros(356, dtype = np.uint8)
        payload[0:6], payload[6:10], payload[10] = ClientHello[0:6], ClientHello[7:11], ClientHello[43]
        payload[11] = (ClientHello[44 + ClientHello[43]]*256 + ClientHello[45 + ClientHello[43]]) // 2
        
        L = 46 + ClientHello[43]
        r = payload[11]*2 if payload[11] <= 35 else 70
        payload[12: 12 + r] =  ClientHello[L:L + r]
        
        L += payload[11]*2 + ClientHello[L + payload[11]*2]+1
        payload[82: 82+2] = ClientHello[L:L+2]
        
        L+=2
        
        extensions = ClientHello[L:]
        ext_order_counter = 85 
        length_of_extensions = payload[82]*256 + payload[83]
        

        seq_dict = {0:303, 21: 111, 35:113, 41:115, 51:133}
        ext_type_dict = {65281: 254, 13172:253, 17513:252}
        
        list_uint_dict = {11: {'data': 0, 'payload': 139, 'R':3},
                            10: {'data': 1, 'payload': 151, 'R':24}, 
                            13: {'data': 1, 'payload': 176, 'R':24},
                            43: {'data': 0, 'payload': 226, 'R':10},
                            16: {'data': 1, 'payload': 305, 'R':50}}
        
        uint8_dict = {3: {'payload': 104, 'data': -1}, 
                    15: {'payload': 104, 'data': -1}, 
                    45: {'payload': 106, 'data': -1}, 
                    17: {'payload': 122, 'data': 0}, 
                    25: {'payload': 124, 'data': 0}}
        uint16_dict = {27: 107, 28: 109, 58:126}
        
        GREASE_ext_counter = 0
        L=0
        while L < length_of_extensions:
            ext_type = extensions[L]*256 + extensions[L+1]
            ext_len  = extensions[L+2]*256 + extensions[L+3]
            ext_data = extensions[L+4:L+4+ext_len]
            L+= ext_len + 4
            
            if ext_order_counter < 104:
                if ext_type <= 59: 
                    payload[237+ext_type] = ext_order_counter - 84
                    #first 60 positions after main payload, paylaod[237] = CH_0_index, payload[296] = CH_59_index
                    #payload[ext_order_counter] = ext_type
                elif ext_type in ext_type_dict:
                    #payload[ext_order_counter] = ext_type_dict[ext_type]
            #        #first three positions after 60 resevered extensions payload[297] = CH_17513_index, 297 = 252 + 45 
                     payload[45 + ext_type_dict[ext_type]] = ext_order_counter - 84
                else: 
                    #payload[300] = GREASE_0, payload[302] = GREASE_2
                    #payload[ext_order_counter] = 255
                    if GREASE_ext_counter < 3:
                        payload[300 + GREASE_ext_counter] = ext_order_counter - 84
                        GREASE_ext_counter+=1
            ext_order_counter+=1

            if ext_type in seq_dict:
                payload[seq_dict[ext_type]:seq_dict[ext_type] + 2] = extensions[L - ext_len - 2:L - ext_len]
                continue
            elif ext_type in uint8_dict:
                payload[uint8_dict[ext_type]['payload']] = ext_data[uint8_dict[ext_type]['data']]
                continue
            elif ext_type in uint16_dict:
                payload[uint16_dict[ext_type]:uint16_dict[ext_type]+2] = ext_data[-2:]
                continue      
            elif ext_type in list_uint_dict:
                N = ext_data[list_uint_dict[ext_type]['data']]
                payload[list_uint_dict[ext_type]['payload']] = N
                R = N if N <= list_uint_dict[ext_type]['R'] else list_uint_dict[ext_type]['R']
                payload[list_uint_dict[ext_type]['payload']+1:list_uint_dict[ext_type]['payload']+1+R] = ext_data[list_uint_dict[ext_type]['data']+1:list_uint_dict[ext_type]['data']+1+R]
                continue
            elif ext_type == 24:
                payload[128:130] = ext_data[:2]
        payload[84] = ext_order_counter - 85
        
        #if GREASE:
        #    SG1 = payload[152]*256 + payload[153]
        #    if not (SG1 <= 41 or (SG1>=256 and SG1<=260) or SG1 == 65281 or SG1 == 65282):
        #        payload[152], payload[153] = 255, 255 
        #    
        #    CS0 = payload[12]*256 +  payload[13]
        #    logic1 = (CS0 >=0 and CS0<=70) or (CS0 >=103 and CS0<=109) or (CS0 >=132 and CS0<=199) or CS0 == 255
        #    logic2 = (CS0 >=4865 and CS0<=4869) or  CS0 == 22016 or (CS0 >=49153 and CS0<=49333)
        #    logic3 = (CS0 >=49408 and CS0<=49414) or (CS0 >=52392 and CS0<=52398) or (CS0 >=53249 and CS0<=53251)
        #    if not (logic1 or logic2 or logic3 or CS0 == 53253):
        #        payload[12], payload[13] = 255,255
        
        return payload

    def SH_short_decomposition(ServerHello: np.array) -> np.array:
        payload = np.zeros(118, dtype = np.uint8)
        #ContentType, Record version, Record Len, Handshake Type, Msg len, Msg Ver, SID len
        payload[0:6], payload[6:10], payload[10] = ServerHello[0:6], ServerHello[7:11], ServerHello[43]
        #cipher
        payload[12:14] = ServerHello[44 + ServerHello[43]:46 + ServerHello[43]]
        #ext_len
        payload[14: 16] = ServerHello[47 + ServerHello[43]:49 + ServerHello[43]]
        
        ext_type_dict = {65281: 254, 13172:253, 17513:252}
        seq_dict = {18: 36, 35:38, 41:40, 51:42}
        
        len_of_extensions = payload[14]*256 + payload[15]
        extensions = ServerHello[49 + ServerHello[43]:]
        L = 0
        
        GREASE_ext_counter = 0
        ext_order_counter = 17 
        
        while L < len_of_extensions:
            ext_type = extensions[L]*256 + extensions[L+1]
            ext_len  = extensions[L+2]*256 + extensions[L+3]
            ext_data = extensions[L+4:L+4+ext_len]
            
            L+= ext_len + 4
            
            if ext_order_counter < 36:
                if ext_type <= 59: 
                    payload[ext_order_counter] = ext_type
                    #payload[52] = SH_0_index, payload[111] = SH_59_index
                    payload[52 + ext_type] = ext_order_counter - 16
                elif ext_type in ext_type_dict:
                    payload[ext_order_counter] = ext_type_dict[ext_type]
                    #payload[112] = SH_17513_index, 112 = 252-140
                    payload[ext_type_dict[ext_type]-140] = ext_order_counter - 16
                else: 
                    payload[ext_order_counter] = 255
                    if GREASE_ext_counter < 3:
                        payload[115 + GREASE_ext_counter] = ext_order_counter - 16
                        GREASE_ext_counter +=1
            ext_order_counter+=1
            
            if ext_type == 43:
                payload[46:48] = ext_data[:2]
                continue
            elif ext_type in seq_dict:
                payload[seq_dict[ext_type]:seq_dict[ext_type]+2] = extensions[L-ext_len-2: L-ext_len]
                if ext_type == 51:
                    payload[44:46] = ext_data[:2]        
            elif ext_type == 11:
                payload[48] = ext_data[0]
                r = payload[48] if payload[48] < 3 else 3
                payload[49:49+r] = ext_data[1:1+r]
        
        payload[16] = ext_order_counter - 17   
        return payload
    
    return np.concatenate([CH_short_decomposition(ClientHello_ServerHello[0],GREASE = GREASE), 
                            SH_short_decomposition(ClientHello_ServerHello[1])])

@vectorizaition(otypes=[np.uint8], signature='()->(n)')
def processed_data(Data: np.array):
    return [CH_payload_decomp(ClientHello_ServerHello[0]), SH_payload_decomp(ClientHello_ServerHello[1])]

def payload_decomposition(ClientHello_ServerHello: np.array):
    assert len(ClientHello_ServerHello) == 2, 'Incorrect Size'
    return [CH_payload_decomp(ClientHello_ServerHello[0]), SH_payload_decomp(ClientHello_ServerHello[1])]

def SH_payload_decomp(ServerHello: np.array) -> np.array:
    payload = np.zeros(237, dtype = np.uint8)
    #ContentType, Record version, Record Len, Handshake Type, Msg len, Msg Ver, SID len
    payload[0:6], payload[6:10], payload[10] = ServerHello[0:6], ServerHello[7:11], ServerHello[43]
    #cipher
    payload[12:14] = ServerHello[44 + ServerHello[43]:46 + ServerHello[43]]
    #ext_len
    payload[82: 84] = ServerHello[47 + ServerHello[43]:49 + ServerHello[43]]
    
    ext_type_dict = {65281: 254, 13172:253, 17513:252}
    seq_dict = {18: 143, 35:113, 41:115, 51:133}
    
    len_of_extensions = payload[82]*256 + payload[83]
    extensions = ServerHello[49 + ServerHello[43]:]
    L = 0
    
    ext_order_counter = 85 
    
    while L < len_of_extensions:
        ext_type = extensions[L]*256 + extensions[L+1]
        ext_len  = extensions[L+2]*256 + extensions[L+3]
        ext_data = extensions[L+4:L+4+ext_len]
        
        L+= ext_len + 4
        
        if ext_order_counter < 104:
            if ext_type <= 59: 
                payload[ext_order_counter] = ext_type
            elif ext_type in ext_type_dict:
                payload[ext_order_counter] = ext_type_dict[ext_type]
            else: 
                payload[ext_order_counter] = 255
        ext_order_counter+=1
        
        if ext_type == 43:
            payload[227:229] = ext_data[:2]
            continue
        elif ext_type in seq_dict:
            payload[seq_dict[ext_type]:seq_dict[ext_type]+2] = extensions[L-ext_len-2: L-ext_len]
            if ext_type == 51:
                payload[152:154] = ext_data[:2]        
        elif ext_type == 11:
            payload[139] = ext_data[0]
            r = payload[139] if payload[139] < 3 else 3
            payload[140:140+r] = ext_data[1:1+r]
    
    payload[84] = ext_order_counter - 85   
    return payload

def CH_payload_decomp(ClientHello: np.array, GREASE = True) -> np.array:
    payload = np.zeros(237, dtype = np.uint8)
    payload[0:6], payload[6:10], payload[10] = ClientHello[0:6], ClientHello[7:11], ClientHello[43]
    payload[11] = (ClientHello[44 + ClientHello[43]]*256 + ClientHello[45 + ClientHello[43]]) // 2
    
    L = 46 + ClientHello[43]
    r = payload[11]*2 if payload[11] <= 35 else 70
    payload[12: 12 + r] =  ClientHello[L:L + r]
    
    L += payload[11]*2 + ClientHello[L + payload[11]*2]+1
    payload[82: 82+2] = ClientHello[L:L+2]
    
    L+=2
    
    extensions = ClientHello[L:]
    ext_order_counter = 85 
    length_of_extensions = payload[82]*256 + payload[83]
    

    seq_dict = {21: 111, 35:113, 41:115, 51:133}
    ext_type_dict = {65281: 254, 13172:253, 17513:252}
    
    list_uint_dict = {11: {'data': 0, 'payload': 139, 'R':3},
                          10: {'data': 1, 'payload': 151, 'R':24}, 
                          13: {'data': 1, 'payload': 176, 'R':24},
                          43: {'data': 0, 'payload': 226, 'R':10}}
    
    uint8_dict = {3: {'payload': 104, 'data': -1}, 
                 15: {'payload': 104, 'data': -1}, 
                 45: {'payload': 106, 'data': -1}, 
                 17: {'payload': 122, 'data': 0}, 
                 25: {'payload': 124, 'data': 0}}
    uint16_dict = {27: 107, 28: 109, 58:126}
    
    L=0
    while L < length_of_extensions:
        ext_type = extensions[L]*256 + extensions[L+1]
        ext_len  = extensions[L+2]*256 + extensions[L+3]
        ext_data = extensions[L+4:L+4+ext_len]
        L+= ext_len + 4
        
        if ext_order_counter < 104:
            if ext_type <= 59: 
                payload[ext_order_counter] = ext_type
            elif ext_type in ext_type_dict:
                payload[ext_order_counter] = ext_type_dict[ext_type]
            else: 
                payload[ext_order_counter] = 255
        ext_order_counter+=1

        if ext_type in seq_dict:
            payload[seq_dict[ext_type]:seq_dict[ext_type] + 2] = extensions[L - ext_len - 2:L - ext_len]
            continue
        elif ext_type in uint8_dict:
            payload[uint8_dict[ext_type]['payload']] = ext_data[uint8_dict[ext_type]['data']]
            continue
        elif ext_type in uint16_dict:
            payload[uint16_dict[ext_type]:uint16_dict[ext_type]+2] = ext_data[-2:]
            continue      
        elif ext_type in list_uint_dict:
            N = ext_data[list_uint_dict[ext_type]['data']]
            payload[list_uint_dict[ext_type]['payload']] = N
            R = N if N <= payload[list_uint_dict[ext_type]['R']] else payload[list_uint_dict[ext_type]['R']]
            payload[list_uint_dict[ext_type]['payload']+1:list_uint_dict[ext_type]['payload']+1+R] = ext_data[1:1+R]
            continue
        elif ext_type == 24:
            payload[128:130] = ext_data[:2]
    payload[84] = ext_order_counter - 85
    
    if GREASE:
        SG1 = payload[152]*256 + payload[153]
        if not (SG1 <= 41 or (SG1>=256 and SG1<=260) or SG1 == 65281 or SG1 == 65282):
            payload[152], payload[153] = 255, 255 
        
        CS0 = payload[12]*256 +  payload[13]
        logic1 = (CS0 >=0 and CS0<=70) or (CS0 >=103 and CS0<=109) or (CS0 >=132 and CS0<=199) or CS0 == 255
        logic2 = (CS0 >=4865 and CS0<=4869) or  CS0 == 22016 or (CS0 >=49153 and CS0<=49333)
        logic3 = (CS0 >=49408 and CS0<=49414) or (CS0 >=52392 and CS0<=52398) or (CS0 >=53249 and CS0<=53251)
        if not (logic1 or logic2 or logic3 or CS0 == 53253):
            payload[12], payload[13] = 255,255
        
    return payload


def balance_train(Train: pd.DataFrame, new_class_power: int, label_column = 'service_label') -> pd.DataFrame:
    
    Train_b  = Train.copy()
    class_names = set(Train_b[label_column])
    for class_name in class_names:
        Train_b_class = Train_b.loc[Train_b[label_column] == class_name]
        class_power = len(Train_b_class)
        print('Label: {0}, Initial class_power: {1}, New class_power: {2}'.format(class_name.center(20, '_'), 
                                                                                  class_power, new_class_power))
        Train_b_class.index = range(class_power)
        for i in range(class_power, new_class_power):
            copied_object = random.randint(0,class_power-1)
            Train_b = Train_b.append(Train_b_class.iloc[copied_object])
    
    return Train_b

def class_distribution(Train: pd.DataFrame, Validation: pd.DataFrame, Test: pd.DataFrame, label_column = 'service_label'):
    
    class_names = set(Train[label_column])
    for class_name in class_names:
        print('{}{} Train: {}, Validation: {}, Test: {}'.format(class_name, ''.join('_' for x in range(20 - len(class_name))),
                                                              len(Train.loc[Train[label_column] == class_name]),
                                                              len(Validation.loc[Validation[label_column] == class_name]),
                                                              len(Test.loc[Test[label_column] == class_name])))


def features_to_tensor(X_features: pd.DataFrame, shape = None) -> tf.Tensor:
    
    def int_to_base256(value: int) -> list:
        return [value//256, value%256]

    X_tensor = list()
    X_features_cp = X_features.copy()
    X_features_cp.index = range(len(X_features_cp.index)) #indexes redefinition
    features_num = len(X_features_cp.iloc[0])
    for i in X_features_cp.index:
        
        object_data = list(map(int_to_base256, np.array(X_features_cp.iloc[i]) + 1))
        if shape is None:
            shape = (1,features_num*2)
            
        object_data = np.array(object_data).reshape(shape)
      
        X_tensor.append(object_data)
    
    X_tensor = np.array(X_tensor, dtype = np.float32)/255
    return tf.convert_to_tensor(X_tensor)

def labels_to_tensor(labels: np.array, label2int_map_dict: dict, one_hot = False) -> tf.Tensor: 
    labels = tf.convert_to_tensor([label2int_map_dict[label] for label in labels])
    if one_hot:
        labels = tf.one_hot(labels, len(label2int_map_dict))
    return labels

def labels_to_int(label_names: list, label2int_map_dict: dict):
    def maplabel2int(label_name):
        return label2int_map_dict[label_name]
    return list(map(maplabel2int, label_names))


def payload_to_tensor(X_payload: pd.DataFrame, columns: list, shape = (18, 150), 
                                            req_payload_len = 900, sep = ',') -> tf.Tensor:

    def padded_payload(PKT_payload: str, req_payload_len: int, sep: str) -> np.array:
        if PKT_payload == '':
            return np.zeros(req_payload_len, dtype = int)
        else:
            try:
                payload_int = list(np.array(PKT_payload.split(sep), dtype = int))
                diff = req_payload_len - len(payload_int)
                padded_payload_int =  payload_int + list(np.zeros(diff)) if diff > 0 else payload_int[:req_payload_len]
                return np.array(padded_payload_int, dtype = np.float32)/255 
            except:  
                return np.zeros(req_payload_len, dtype = int)

    
    data = list()
    X_payload.index = range(len(X_payload.index)) #indexes redefinition
    for i in X_payload.index:
        object_data = list()
        for column in columns:
            object_data += list(padded_payload(X_payload.iloc[i][column], 
                                              req_payload_len = req_payload_len, sep = sep))
        
        object_data = list(np.reshape(object_data, shape))
        data.append(object_data)
        
    return tf.convert_to_tensor(data)

def CH_SH_to_tensor(X_payload: pd.DataFrame, units = 256, CH_ext = 2, 
                                               SH_ext = 1, sep = ',') -> tf.Tensor:
    
    def trunc_pad_payload(payload_int: list, req_payload_len: int) -> np.array:
        diff = req_payload_len - len(payload_int)
        padded_payload_int =  payload_int + list(np.zeros(diff)) if diff > 0 else payload_int[:req_payload_len]
        return np.array(padded_payload_int, dtype = np.float32)/255 

    def SH_fields_length(ServerHello: np.array) -> int:
        return ServerHello[43] + 47

    def CH_fields_length(ClientHello: np.array) -> int:
        Sid_len = ClientHello[43]
        Ciphers_Len = ClientHello[44 + Sid_len: 46 + Sid_len][0]*256 + ClientHello[44 + Sid_len: 46 + Sid_len][1]
        Complen = ClientHello[46 + Sid_len + Ciphers_Len]
        return 47 + Sid_len + Ciphers_Len + Complen

    def ClientHello_split(ClientHello_s: str, units = 256, 
                                                  ext_parts = 2, sep = ','):
            x = ext_parts+1
            if ClientHello_s == '':
                return list(np.zeros(x*units, dtype = int))
            else:
                try:
                    ClientHello = np.array(ClientHello_s.split(sep), dtype = int)
                    CH_fields_len = CH_fields_length(ClientHello)
                    CH_fields, CH_extensions = ClientHello[:CH_fields_len], ClientHello[CH_fields_len:]
                    CH_fields = trunc_pad_payload(list(CH_fields), req_payload_len = units)
                    CH_extensions = trunc_pad_payload(list(CH_extensions), req_payload_len = ext_parts*units)
                    return list(CH_fields)+list(CH_extensions)
                except:  
                    return list(np.zeros(x*units, dtype = int))

    def ServerHello_split(ServerHello_s: str, units = 256, 
                                                  ext_parts = 1, sep = ','):
            x = ext_parts+1
            if ServerHello_s == '':
                return list(np.zeros(x*units, dtype = int))
            else:
                try:
                    ServerHello = np.array(ServerHello_s.split(sep), dtype = int)
                    SH_fields_len = SH_fields_length(ServerHello)
                    SH_fields, SH_extensions = ServerHello[:SH_fields_len], ServerHello[SH_fields_len:]
                    SH_fields = trunc_pad_payload(list(SH_fields), req_payload_len = units)
                    SH_extensions = trunc_pad_payload(list(SH_extensions), req_payload_len = ext_parts*units)
                    return list(SH_fields)+list(SH_extensions)
                except:  
                    return list(np.zeros(x*units, dtype = int))

    data = list()
    X_payload.index = range(len(X_payload.index)) #indexes redefinition
    for i in X_payload.index:
        object_data = ClientHello_split(X_payload.iloc[i]['PKT_1_payload'], ext_parts = CH_ext)
        object_data += ServerHello_split(X_payload.iloc[i]['PKT_2_SH_only_payload'], ext_parts = SH_ext)    
        object_data = list(np.reshape(object_data, (2+CH_ext+SH_ext,units)))
        data.append(object_data)
        
    return tf.convert_to_tensor(data)

def mapdict_constructor(snis_set) -> dict:
    snis_set = set(snis_set)
    sorted_sni = sorted(list(snis_set), key = lambda x: x.split('.')[-2])
    mapdict = {}
    for index, sni in enumerate(sorted_sni):
        mapdict[sni] = index
    del sorted_sni
    return mapdict

def mapdict_from_label_names(y):
    classes = sorted(list(set(y)))
    mapdict = {}
    counter = 0
    for class_ in classes:
        mapdict[class_] = counter
        counter+=1
    return mapdict


class Dataset_threshold_variations():
    
    def __init__(self, read_csv_params, min_flows_num_in_class = range(1000, 3000, 500), 
                        main_domains = True, only_letters = True, google_unite = False):
        self.Dataset = pd.read_csv(**read_csv_params)
        self.main_domains = main_domains
        self.only_letters = only_letters
        self.google_unite = google_unite
        self.thresholds = min_flows_num_in_class
        self.Dataset['SNI_label']  = self.processed_snis_constructor()
        self.threshold_params = self.threshold_functions_constructor()
        
    def processed_snis_constructor(self) -> list:
        
        def sni_main_domains(server_name: str) -> str:
            parts = server_name.split('.')
            if len(parts) <= 2:
                output = server_name
            elif len(parts) <= 4:
                output =  '.'.join(parts[1:])
            else:
                output =  '.'.join(parts[-4:])
            del parts
            return output

        def sni_only_letters(server_name: str) -> str:
            processed_sni = ''
            for symbol in server_name:
                if symbol.isalpha() or symbol == '.':
                    processed_sni+=symbol
            return processed_sni

        def sni_google_unite(server_name: str) -> str:
            if server_name.find('google') >= 0 or server_name.find('gstatic') >=0 or  server_name.find('youtube') >=0:
                return 'google.united'
            else:
                return server_name
        
        print('Constructing labels')
        
        server_names = list(self.Dataset.SNI)
        processed_server_names = []
        for i, sni in tqdm(enumerate(server_names)):
            processed_sni = sni
            processed_sni = sni_main_domains(processed_sni) if self.main_domains else processed_sni
            processed_sni = sni_only_letters(processed_sni) if self.only_letters else processed_sni
            processed_sni = sni_google_unite(processed_sni) if self.google_unite else processed_sni
            processed_server_names.append(processed_sni)
        del server_names
        return processed_server_names
    
    def threshold_functions_constructor(self) -> dict:
        
        def sni_set_mask(sni_list, sni_set) -> list:
            return [sni in sni_set for sni in sni_list]

        def mapdict_constructor(snis_set) -> dict:
            sorted_sni = sorted(list(snis_set), key = lambda x: x.split('.')[-2])
            mapdict = {}
            for index, sni in enumerate(sorted_sni):
                mapdict[sni] = index
            del sorted_sni
            return mapdict
        
        
        processed_server_names = list(self.Dataset['SNI_label'])
        
        processed_sni_counts = dict()
        for sni in set(processed_server_names):
            processed_sni_counts[sni] = processed_server_names.count(sni)

        print('Constructing treshold subsets')

        SNIs_for_thresholds = {'count': {}, 'set': {}, 'mask': {}, 'mapdict': {}}

        for t in tqdm(self.thresholds):
            SNIs_for_thresholds['set'][t] = [] 
            for sni in set(processed_server_names):
                if processed_sni_counts[sni] >= t:
                    SNIs_for_thresholds['set'][t].append(sni)

            SNIs_for_thresholds['count'][t] = len(SNIs_for_thresholds['set'][t])

            SNIs_for_thresholds['mask'][t] = sni_set_mask(processed_server_names, 
                                                            SNIs_for_thresholds['set'][t])
            SNIs_for_thresholds['mapdict'][t] =  mapdict_constructor(SNIs_for_thresholds['set'][t])

        del processed_sni_counts, processed_server_names

        return SNIs_for_thresholds
    
    def get_mapdict(self, T: int):
        assert T in set(self.thresholds), 'T not in Tresholds'
        return self.threshold_params['mapdict'][T]
    
    def get_original_Dataset(self):
        return self.Dataset
    
    def get_Dataset_with_threshold(self, T: int):
        assert T in set(self.thresholds), 'T not in Tresholds'
        return self.Dataset.loc[self.threshold_params['mask'][T]]

    def get_class_num(self, T: int):
        assert T in set(self.thresholds), 'T not in Tresholds'
        return self.threshold_params['count'][T]


class Dataset_splitted():
    def __init__(self, Dataset: pd.DataFrame, mapdict: dict, label_column = 'SNI_label'):
        self.mapdict = mapdict
        self.label_column = label_column
        train_validation, self.Test = train_test_split(Dataset, test_size=0.2)
        self.Train, self.Validation = train_test_split(train_validation, test_size=0.1)
        del train_validation
        
    def get_features(self, columns):
        X_train_features = self.Train[columns]
        X_valid_features = self.Validation[columns]
        X_test_features  = self.Test[columns]

        y_train_int = labels_to_int(self.Train[self.label_column], self.mapdict)
        y_valid_int = labels_to_int(self.Validation[self.label_column], self.mapdict)
        y_test_int  = labels_to_int(self.Test[self.label_column], self.mapdict)

        return X_train_features, X_valid_features, X_test_features, y_train_int, y_valid_int, y_test_int
    
    def get_payload(self, columns, shape = (18, 150), req_payload_len = 900):
        X_train_payload = payload_to_tensor(self.Train, columns = columns, shape = shape, req_payload_len = req_payload_len)
        X_valid_payload = payload_to_tensor(self.Validation, columns = columns, shape = shape, req_payload_len = req_payload_len)
        X_test_payload  = payload_to_tensor(self.Test, columns = columns, shape = shape, req_payload_len = req_payload_len)

        y_train_one_hot = labels_to_tensor(self.Train[self.label_column], self.mapdict, one_hot = True)
        y_valid_one_hot = labels_to_tensor(self.Validation[self.label_column], self.mapdict, one_hot = True)
        y_test_one_hot  = labels_to_tensor(self.Test[self.label_column], self.mapdict, one_hot = True)

        return X_train_payload, X_valid_payload, X_test_payload, y_train_one_hot, y_valid_one_hot, y_test_one_hot
    
    def get_CH_SH_payload(self, units = 256):
        
        X_train_CH_SH_payload = CH_SH_to_tensor(self.Train, units = units)
        X_valid_CH_SH_payload = CH_SH_to_tensor(self.Validation, units = units)
        X_test_CH_SH_payload  = CH_SH_to_tensor(self.Test, units = units)

        y_train_one_hot = labels_to_tensor(self.Train[self.label_column], self.mapdict, one_hot = True)
        y_valid_one_hot = labels_to_tensor(self.Validation[self.label_column], self.mapdict, one_hot = True)
        y_test_one_hot  = labels_to_tensor(self.Test[self.label_column], self.mapdict, one_hot = True)

        return X_train_CH_SH_payload, X_valid_CH_SH_payload, X_test_CH_SH_payload, y_train_one_hot, y_valid_one_hot, y_test_one_hot

def Data_threshold(Data:np.array, labels: np.array, threshold: int):
    assert len(Data) == len(labels), 'labels and Data have inappropriate sizes'
    count_dict = {}
    for item in set(labels):
        count_dict[item] = list(labels).count(item)
    Data_new = []
    labels_new = [] 
    for i in range(len(Data)):
        if count_dict[labels[i]] >= threshold:
            Data_new.append(Data[i])
            labels_new.append(labels[i])
    return np.array(Data_new), np.array(labels_new)

def zero_random_utf(Hello: np.array):
    Hello[11:15] = np.zeros(4, dtype = np.uint8)
    return Hello

def EnryptedCH_payload_(ClientHello: np.array, hard_ECH = False, change_len = True) -> np.array:
    CS_len = ClientHello[44 + ClientHello[43]]*256 + ClientHello[45 + ClientHello[43]]
     
    CS_end = 46 + ClientHello[43] + CS_len
    CompMet_len = ClientHello[CS_end]
    ext_len_start = CS_end + CompMet_len + 1
    length_of_extensions = ClientHello[ext_len_start]*256 + ClientHello[ext_len_start+1]
    extensions = ClientHello[ext_len_start+2: ]
    
    new_extensions = np.array([], dtype = np.uint8)
    length_of_delete_extensions = 0
    L=0
    while L < length_of_extensions:
        
        ext_type = extensions[L]*256 + extensions[L+1]
        ext_len  = extensions[L+2]*256 + extensions[L+3]
        if (not hard_ECH and not ext_type in set([0,16])) or (hard_ECH and ext_type in set([51,43,41])):
            new_extensions = np.concatenate([new_extensions, extensions[L:L+4+ext_len]])
        else: 
            length_of_delete_extensions += ext_len+4
        
        L+= 4+ext_len
        
    new_length_of_extensions = np.array([(length_of_extensions - length_of_delete_extensions)//256, 
                                        (length_of_extensions - length_of_delete_extensions)%256], dtype = np.uint8)    
    
    ECH_pay = np.concatenate([ClientHello[:ext_len_start],  new_length_of_extensions, new_extensions])
    
    if change_len:
        new_rec_len = ClientHello[3]*256 + ClientHello[4] - length_of_delete_extensions
        ECH_pay[3], ECH_pay[4] = new_rec_len//256, new_rec_len%256
        ECH_pay[7], ECH_pay[8] = (new_rec_len-4)//256, (new_rec_len-4)%256
    
    return ECH_pay

def EnryptedCH_payload(ClientHello: np.array, change_len = True, encrypted_extensions = set([0,16])) -> np.array:
    CS_len = ClientHello[44 + ClientHello[43]]*256 + ClientHello[45 + ClientHello[43]]
     
    CS_end = 46 + ClientHello[43] + CS_len
    CompMet_len = ClientHello[CS_end]
    ext_len_start = CS_end + CompMet_len + 1
    length_of_extensions = ClientHello[ext_len_start]*256 + ClientHello[ext_len_start+1]
    extensions = ClientHello[ext_len_start+2: ]
    
    new_extensions = np.array([], dtype = np.uint8)
    length_of_delete_extensions = 0
    L=0
    while L < length_of_extensions:
        
        ext_type = extensions[L]*256 + extensions[L+1]
        ext_len  = extensions[L+2]*256 + extensions[L+3]
        if (not ext_type in encrypted_extensions):
            new_extensions = np.concatenate([new_extensions, extensions[L:L+4+ext_len]])
        else: 
            length_of_delete_extensions += ext_len+4
        
        L+= 4+ext_len
        
    new_length_of_extensions = np.array([(length_of_extensions - length_of_delete_extensions)//256, 
                                        (length_of_extensions - length_of_delete_extensions)%256], dtype = np.uint8)    
    
    ECH_pay = np.concatenate([ClientHello[:ext_len_start],  new_length_of_extensions, new_extensions])
    
    if change_len:
        new_rec_len = ClientHello[3]*256 + ClientHello[4] - length_of_delete_extensions
        ECH_pay[3], ECH_pay[4] = new_rec_len//256, new_rec_len%256
        ECH_pay[7], ECH_pay[8] = (new_rec_len-4)//256, (new_rec_len-4)%256
    
    return ECH_pay

def CH_payload_with_plastered_SNI(ClientHello: np.array) -> np.array:
    CS_len = ClientHello[44 + ClientHello[43]]*256 + ClientHello[45 + ClientHello[43]]
     
    CS_end = 46 + ClientHello[43] + CS_len
    CompMet_len = ClientHello[CS_end]
    ext_len_start = CS_end + CompMet_len + 1
    length_of_extensions = ClientHello[ext_len_start]*256 + ClientHello[ext_len_start+1]
    extensions = ClientHello[ext_len_start+2: ]

    L=0
    while L < length_of_extensions:
        
        ext_type = extensions[L]*256 + extensions[L+1]
        ext_len  = extensions[L+2]*256 + extensions[L+3]
        if ext_type == 0:
            ClientHello[ext_len_start+2+L+4:ext_len_start+2+L+4+ext_len] = np.zeros(ext_len)
        L+= 4+ext_len

    return ClientHello

@vectorizaition(otypes=[object], signature='()->()')
def SNI_extraction_vector(ClientHello: np.array):
    return SNI_extraction(ClientHello)

def SNI_extraction_parallel(ClientHellos: list, njobs = 1) -> np.array:
    n_proc = range(cpu_count()+1)[njobs]
    with Pool(processes=n_proc) as pool:
        #result = pool.map(payload_decomposition, Data) if one_vector else pool.map(CH_and_SH_decomposition, Data)
        result = pool.map(SNI_extraction, ClientHellos, chunksize=len(ClientHellos) // n_proc)
    return result

def SNI_extraction(ClientHello: np.array):
    CS_len = ClientHello[44 + ClientHello[43]]*256 + ClientHello[45 + ClientHello[43]]
     
    CS_end = 46 + ClientHello[43] + CS_len
    CompMet_len = ClientHello[CS_end]
    ext_len_start = CS_end + CompMet_len + 1
    length_of_extensions = ClientHello[ext_len_start]*256 + ClientHello[ext_len_start+1]
    extensions = ClientHello[ext_len_start+2: ]

    L=0
    while L < length_of_extensions:
        
        ext_type = extensions[L]*256 + extensions[L+1]
        ext_len  = extensions[L+2]*256 + extensions[L+3]
        if ext_type == 0:
            #5 additional bytes on SNI type and two internal length fields 
            #return ClientHello[ext_len_start+2+L+4+5:ext_len_start+2+L+4+ext_len]
            return ''.join(bytes_to_string(ClientHello[ext_len_start+2+L+4+5:ext_len_start+2+L+4+ext_len] ))
        L+= 4+ext_len

    return ''



def ClientHellos_from_dataset(Dataset: pd.DataFrame) -> list:
    CHs = []
    Dataset.index = range(len(Dataset))
    for i in tqdm(Dataset.index):
         try:
            pkt1 = np.array(Dataset['PKT_1_payload'][i].split(','), dtype = np.uint8)
            if (pkt1[0] == 22 and pkt1[5] == 1):
                CHs.append(pkt1)
         except:
            continue
    
    return CHs

def SH_without_EE_payload(ServerHello: np.array, change_len = True, EE_types = set([0,1,10,14,15,16,19,20,28,32,39,42,55,56,57,58])) -> np.array:
    length_of_extensions = ServerHello[47 + ServerHello[43]]*256 + ServerHello[48 + ServerHello[43]]
    extensions = ServerHello[49 + ServerHello[43]:]
    
    new_extensions = np.array([], dtype = np.uint8)
    length_of_delete_extensions = 0
    
    L=0
    while L < length_of_extensions:
        ext_type = extensions[L]*256 + extensions[L+1]
        ext_len  = extensions[L+2]*256 + extensions[L+3]
        
        if not ext_type in EE_types:
            new_extensions = np.concatenate([new_extensions, extensions[L:L+4+ext_len]])
        else: 
            length_of_delete_extensions += ext_len+4    
        #mystery oscp byte
        if (ext_type == 5 and ext_len == 0) and (extensions[L+4] == 0 and extensions[L+5] == 0):
            L+=1 
        L+= 4+ext_len
        
    new_length_of_extensions = np.array([(length_of_extensions - length_of_delete_extensions)//256, 
                                        (length_of_extensions - length_of_delete_extensions)%256], dtype = np.uint8)    
    
    SH_pay = np.concatenate([ServerHello[:47 + ServerHello[43]], new_length_of_extensions,
                            new_extensions])
        
        
    if change_len:
        new_rec_len = ServerHello[3]*256 + ServerHello[4] - length_of_delete_extensions
        SH_pay[3], SH_pay[4] = new_rec_len//256, new_rec_len%256
        SH_pay[7], SH_pay[8] = (new_rec_len-4)//256, (new_rec_len-4)%256
    
        
    return SH_pay


def payload_data_exctraction(Dataset: pd.DataFrame, label_column: str, scenario = 'hardECH', 
                            ech_encrypted_extensions = set(list(range(41)) + [42] + list(range(44,51)) + list(range(52,61)) + [65281]), EE_types = set([0,1,10,14,15,16,19,20,28,32,39,42,55,56,57,58])):
    
    assert scenario == 'hardECH' or scenario == 'lightECH' or scenario == 'TLS1.2' or scenario == 'TLS1.3', 'Scenario is not correct!'
    EncryptedCH = True if scenario == 'hardECH' or scenario == 'lightECH'  else False
    EncryptedExtensions = True if scenario == 'TLS1.3' or scenario == 'hardECH' or scenario == 'lightECH' else False
    ThirdPacket = True if scenario == 'TLS1.2' else False
    change_len = True if scenario == 'hardECH' else False

    print('Initial Dataset Size: {}'.format(len(Dataset)))
    print('Only flows with the correct handshake remain')
    print('Payload conversion from string to uint8 array')
    if EncryptedCH:
        print('ClientHello payload without SNI and ALPN')
    if EncryptedExtensions:
        print('ServerHello payload without extensions marked as "enrypted" by IANA')
    if change_len:
        print('Length of ClientHello has changed')
    
    Data, labels = [], []

    Dataset.index = range(len(Dataset))
    for i in tqdm(Dataset.index):
         try:
            pkt1 = np.array(Dataset['PKT_1_payload'][i].split(','), dtype = np.uint8)
            pkt2 = np.array(Dataset['PKT_2_payload'][i].split(','), dtype = np.uint8)
            #Check that pkt1 == ClientHello and pkt2 == ServerHello
            if (pkt1[0] == 22 and pkt1[5] == 1) and (pkt2[0] == 22 and pkt2[5] == 2):
                pkt1 = CH_payload_with_plastered_SNI(zero_random_utf(pkt1))
                pkt2 = zero_random_utf(pkt2)
                if EncryptedCH:
                    pkt1 = EnryptedCH_payload(pkt1, change_len=change_len, encrypted_extensions=ech_encrypted_extensions)
                if EncryptedExtensions:
                    pkt2 = SH_without_EE_payload(pkt2, EE_types)
                if ThirdPacket:
                    try: 
                        pkt3 = np.array(Dataset['PKT_3_payload'][i].split(','), dtype = np.uint8)
                    except:
                        pkt3 = np.array([], dtype = np.uint8)
                    Data.append([pkt1, pkt2, pkt3])
                else:
                    Data.append([pkt1, pkt2])
                labels.append(Dataset[label_column][i])
         except:
             continue

    print('The resulting Dataset Size: {}'.format(len(Data)))
    return np.array(Data, dtype=object), np.array(labels, dtype=object)




               

def dataset_rec_ext_sni_pad_lengths(Dataset: pd.DataFrame):
    mes_sni_pad = {'SNI': [], 'mes_len': [], 'ext_len': [], 'sni_len': [], 'pad_len': []}

    Dataset.index = range(len(Dataset))
    for i in tqdm(Dataset.index):
        try:
            pkt1 = np.array(Dataset['PKT_1_payload'][i].split(','), dtype = np.uint8)
            if (pkt1[0] == 22 and pkt1[5] == 1):
                rec_len, ext_len, sni_len, pad_len = find_rec_ext_sni_pad_lengths(pkt1)
                mes_sni_pad['SNI'].append(Dataset['SNI'][i])
                mes_sni_pad['mes_len'].append(rec_len)
                mes_sni_pad['ext_len'].append(ext_len)
                mes_sni_pad['sni_len'].append(sni_len)
                mes_sni_pad['pad_len'].append(pad_len)
        except:
            continue

    mes_sni_pad = pd.DataFrame(mes_sni_pad)
    mes_sni_pad['service_label'] = service_label(mes_sni_pad['SNI'])

    return mes_sni_pad
        
def find_rec_ext_sni_pad_lengths(ClientHello: np.array) -> int:
    CS_len = ClientHello[44 + ClientHello[43]]*256 + ClientHello[45 + ClientHello[43]]
     
    rec_len = ClientHello[3]*256 + ClientHello[4]

    CS_end = 46 + ClientHello[43] + CS_len
    CompMet_len = ClientHello[CS_end]
    ext_len_start = CS_end + CompMet_len + 1

    length_of_extensions = ClientHello[ext_len_start]*256 + ClientHello[ext_len_start+1]

    extensions = ClientHello[ext_len_start+2: ]
    
    new_extensions = np.array([], dtype = np.uint8)
    L=0

    sni_len, pad_len = 0,0
    while L < length_of_extensions:
        
        ext_type = extensions[L]*256 + extensions[L+1]
        ext_len  = extensions[L+2]*256 + extensions[L+3]
        if ext_type == 0:
            sni_len = ext_len
        if ext_type == 21:
            pad_len = ext_len
        L+= 4+ext_len
    
    return rec_len, length_of_extensions, sni_len, pad_len


def CH_dict(ClientHello: np.array) -> dict:
    CH = {'ContentType': ClientHello[0],  
          'RecordVersion': ClientHello[1:3],
          'RecordLen': ClientHello[3:5], 
          'HandshakeType': ClientHello[5], 
          'MsgLen': ClientHello[6:9],
          'MsgVersion': ClientHello[9:11],
          'Random': ClientHello[11:43],
          'SID_Len': ClientHello[43],
          'SID': ClientHello[44: 44 + ClientHello[43]],
          'CiphersLen': ClientHello[44 + ClientHello[43]: 46 + ClientHello[43]]}
     
    R = 46 + ClientHello[43] + CH['CiphersLen'][0]*256 + CH['CiphersLen'][1]
    CH['Ciphers'] = ClientHello[46 + ClientHello[43]: R]
    CH['CompMetLen'] = ClientHello[R]
    L = R + 1
    R += CH['CompMetLen']+1
    CH['CompMet'] = ClientHello[L:R]
    L = R
    R+=2
    CH['ExtLen'] = ClientHello[L:R]
    L=R
    R+=2
    ext_len = CH['ExtLen'][0]*256 + CH['ExtLen'][1]
    CH['Extensions'] = extensions_preprocessing(ClientHello[L:], ext_len)
    return CH

def extensions_preprocessing(extensions, ext_len):
    counter = 0 
    ext_dict = {}
    L,R = 0,2
    while R < ext_len:
        ext_payload = {}
        ext_payload['type'] = extensions[L:R]
        #if ext_payload['type'][0] == 0 and ext_payload['type'][1] == 5:
        #    print(extensions[L:L+6])
        L=R
        R+=2
        ext_payload['len'] = extensions[L:R]
        L=R
        R+= 256*ext_payload['len'][0] + ext_payload['len'][1]
        ext_payload['data'] = extensions[L:R]
        L=R
        R+=2
        ext_dict['extension_{}'.format(counter)] = ext_payload
        counter +=1
    return ext_dict

def SH_dict(ServerHello: np.array) -> dict:
    SH = {'ContentType': ServerHello[0],  
          'RecordVersion': ServerHello[1:3],
          'RecordLen': ServerHello[3:5], 
          'HandshakeType': ServerHello[5], 
          'MsgLen': ServerHello[6:9],
          'MsgVersion': ServerHello[9:11],
          'Random': ServerHello[11:43],
          'SID_len': ServerHello[43],
          'SID': ServerHello[44: 44 + ServerHello[43]],
          'Cipher': ServerHello[44 + ServerHello[43]:46 + ServerHello[43]],
          'CompMet': ServerHello[46 + ServerHello[43]]}
    
    SH['ExtLen'] = ServerHello[47 + ServerHello[43]:49 + ServerHello[43]]
    ext_len = ServerHello[47 + ServerHello[43]]*256 + ServerHello[48 + ServerHello[43]]
    SH['Extensions'] = extensions_preprocessing(ServerHello[49 + ServerHello[43]:], ext_len)
    return SH

def Int256toInt10(int256array: np.array) -> int:
    int10 = 0
    size = len(int256array)
    for i in range(size):
        int10+= (256 ** (size - i - 1))*int256array[i]
    del size
    return int10

def int_payload_dict(payload_dict: dict) -> dict:    
    payload_int_dict = payload_dict.copy()
    for field in ['RecordVersion', 'RecordLen', 'MsgLen', 'MsgVersion', 'CiphersLen', 'ExtLen']:
        payload_int_dict[field] = Int256toInt10(payload_dict[field])
    Ciphers = []
    for i in range(len(payload_dict['Ciphers'])//2):
        Ciphers.append(Int256toInt10(payload_dict['Ciphers'][2*i:2*i+2]))
    payload_int_dict['Ciphers'] = np.array(Ciphers)
    for ext in payload_dict['Extensions']:
        payload_int_dict['Extensions'][ext]['type'] = Int256toInt10(payload_dict['Extensions'][ext]['type'])
        payload_int_dict['Extensions'][ext]['len'] = Int256toInt10(payload_dict['Extensions'][ext]['len'])
        
    return payload_int_dict

def view_dict(payload_int_dict: dict) -> dict:
    with open('parameters_names/CipherSuites.json', "r") as fp:
        CipherSuites_names = json.load(fp) 

    with open('parameters_names/ext_names.json', "r") as fp:
        ext_names = json.load(fp) 
    with open('parameters_names/supported_groups_names.json', "r") as fp:
        sg_names = json.load(fp) 
    payload_view_dict = payload_int_dict.copy()
    CiphersName = []
    for cipher in payload_int_dict['Ciphers']:
        if str(cipher) in CipherSuites_names:
            CiphersName.append(CipherSuites_names[str(cipher)])
        else:
            CiphersName.append('GREASE')
    payload_view_dict['Ciphers'] = np.array(CiphersName)
    new_extensions = {}
    for ext in payload_int_dict['Extensions']:
        if payload_int_dict['Extensions'][ext]['type'] == 51:
            data = payload_int_dict['Extensions'][ext]['data']
            key_share_data = {'key share len': Int256toInt10(data[0:2])}
            counter_51 = 0 
            pointer = 2 
            while pointer < payload_int_dict['Extensions'][ext]['len']:
                key_len = Int256toInt10(data[pointer+2:pointer+4])
                key_share_data['Key_{}'.format(counter_51)] = {
                    'Group':   Int256toInt10(data[pointer:pointer+2]), 
                    'key_len': key_len,
                    'key':     data[pointer+4:pointer+4+key_len]
                }
                pointer += 4+key_len
                counter_51+=1
            payload_int_dict['Extensions'][ext]['data'] = key_share_data
        if str(payload_int_dict['Extensions'][ext]['type']) in ext_names:
            payload_view_dict['Extensions'][ext]['type'] = ext_names[str(payload_int_dict['Extensions'][ext]['type'])] + ' ({})'.format(payload_int_dict['Extensions'][ext]['type']) 
        else:
            payload_view_dict['Extensions'][ext]['type'] = 'GREASE' + ' ({})'.format(payload_int_dict['Extensions'][ext]['type']) 
    return payload_view_dict

def CH_view(ClientHello: np.array):
    return view_dict(int_payload_dict(CH_dict(ClientHello)))


def payload_data_exctraction_with_PS_features(Flow_Dataset: np.array, Packet_Dataset: pd.DataFrame, label_column: str, scenario = 'hardECH', 
                            ech_encrypted_extensions = set([0,16,21]), EE_types = set([0,1,10,14,15,16,19,20,28,32,39,42,55,56,57,58])):
    
    Dataset = Packet_Dataset.copy()
    
    assert scenario == 'hardECH' or scenario == 'lightECH' or scenario == 'TLS1.2' or scenario == 'TLS1.3', 'Scenario is not correct!'
    EncryptedCH = True if scenario == 'hardECH' or scenario == 'lightECH'  else False
    EncryptedExtensions = True if scenario == 'TLS1.3' or scenario == 'hardECH' or scenario == 'lightECH' else False
    ThirdPacket = True if scenario == 'TLS1.2' else False
    change_len = True if scenario == 'hardECH' else False

    print('Initial Dataset Size: {}'.format(len(Dataset)))
    print('Only flows with the correct handshake remain')
    print('Payload conversion from string to uint8 array')
    if EncryptedCH:
        print('ClientHello payload without SNI and ALPN')
    if EncryptedExtensions:
        print('ServerHello payload without extensions marked as "enrypted" by IANA')
    if change_len:
        print('Length of ClientHello has changed')
    
    Data, labels = [], []
    
    Dataset.index = range(len(Dataset))
    for i in tqdm(Dataset.index):
         try:
            pkt1 = np.array(Dataset['PKT_1_payload'][i].split(','), dtype = np.uint8)
            pkt2 = np.array(Dataset['PKT_2_payload'][i].split(','), dtype = np.uint8)
            #Check that pkt1 == ClientHello and pkt2 == ServerHello
            if (pkt1[0] == 22 and pkt1[5] == 1) and (pkt2[0] == 22 and pkt2[5] == 2):
                pkt1 = CH_payload_with_plastered_SNI(zero_random_utf(pkt1))
                pkt2 = zero_random_utf(pkt2)
                if EncryptedCH:
                    pkt1 = EnryptedCH_payload(pkt1, change_len = change_len, encrypted_extensions = ech_encrypted_extensions)
                if EncryptedExtensions:
                    pkt2 = SH_without_EE_payload(pkt2, EE_types)
                if ThirdPacket:
                    try: 
                        pkt3 = np.array(Dataset['PKT_3_payload'][i].split(','), dtype = np.uint8)
                    except:
                        pkt3 = np.array([], dtype = np.uint8)
                    Data.append([pkt1, pkt2, pkt3, Flow_Dataset[i]])
                else:
                    Data.append([pkt1, pkt2, Flow_Dataset[i]])
                labels.append(Dataset[label_column][i])
         except:
             continue

    print('The resulting Dataset Size: {}'.format(len(Data)))
    return np.array(Data, dtype=object), np.array(labels, dtype=object)

def get_msg_len(msg):
    return msg[4]+256*msg[3]+5

def get_msg_len_UDP(msg):
    return msg[8]+256*msg[7]+9

def payload_data_and_TSPS_exctraction(Dataset: pd.DataFrame, label_column: str, TSPS_features_columns: np.array, scenario = 'hardECH', 
                            ech_encrypted_extensions = set(list(range(41)) + [42] + list(range(44,51)) + list(range(52,61)) + [65281]), EE_types = set([0,1,10,14,15,16,19,20,28,32,39,42,55,56,57,58])):
    

    Flow_Dataset = Dataset[TSPS_features_columns].to_numpy()

    assert scenario == 'hardECH' or scenario == 'lightECH' or scenario == 'TLS1.2' or scenario == 'TLS1.3', 'Scenario is not correct!'
    EncryptedCH = True if scenario == 'hardECH' or scenario == 'lightECH'  else False
    EncryptedExtensions = True if scenario == 'TLS1.3' or scenario == 'hardECH' or scenario == 'lightECH' else False
    ThirdPacket = True if scenario == 'TLS1.2' else False
    change_len = True if scenario == 'hardECH' else False

    print('Initial Dataset Size: {}'.format(len(Dataset)))
    print('Only flows with the correct handshake remain')
    print('Payload conversion from string to uint8 array')
    if EncryptedCH:
        print('ClientHello payload without SNI and ALPN')
    if EncryptedExtensions:
        print('ServerHello payload without extensions marked as "enrypted" by IANA')
    if change_len:
        print('Length of ClientHello has changed')
    


    Data, labels = [], []
    
    Dataset.index = range(len(Dataset))
    for i in tqdm(Dataset.index):
         try:
            pkt1 = np.array(Dataset['PKT_1_payload'][i].split(','), dtype = np.uint8)
            pkt2 = np.array(Dataset['PKT_2_payload'][i].split(','), dtype = np.uint8)
            #Check that pkt1 == ClientHello and pkt2 == ServerHello
            if (pkt1[0] == 22 and pkt1[5] == 1) and (pkt2[0] == 22 and pkt2[5] == 2):
                if not EncryptedCH:
                    pkt1 = CH_payload_with_plastered_SNI(zero_random_utf(pkt1))
                pkt2 = zero_random_utf(pkt2)
                if EncryptedCH:
                    pkt1 = EnryptedCH_payload(pkt1, change_len = change_len, encrypted_extensions = ech_encrypted_extensions)
                if EncryptedExtensions:
                    pkt2 = SH_without_EE_payload(pkt2, EE_types)
                if ThirdPacket:
                    try: 
                        pkt3 = np.array(Dataset['PKT_3_payload'][i].split(','), dtype = np.uint8)
                    except:
                        pkt3 = np.array([], dtype = np.uint8)
                    Data.append([pkt1, pkt2, pkt3, Flow_Dataset[i]])
                    labels.append(Dataset[label_column][i])
                else:
                    if (get_msg_len(pkt1) == len(pkt1)) and (get_msg_len(pkt2) == len(pkt2)):
                        Data.append([pkt1, pkt2, Flow_Dataset[i]])
                        labels.append(Dataset[label_column][i])
         except:
             continue

    print('The resulting Dataset Size: {}'.format(len(Data)))
    return np.array(Data, dtype=object), np.array(labels, dtype=object)


def payload_data_TSPS_IPdst_exctraction(Dataset: pd.DataFrame, label_column: str, TSPS_features_columns: np.array, IPdstColumn = 'ipDst', scenario = 'hardECH', 
                            ech_encrypted_extensions = set(list(range(41)) + [42] + list(range(44,51)) + list(range(52,61)) + [65281]), EE_types = set([0,1,10,14,15,16,19,20,28,32,39,42,55,56,57,58])):
    

    Flow_Dataset = Dataset[TSPS_features_columns].to_numpy()
    ipDst_Dataset = ipVector(Dataset[IPdstColumn])

    assert scenario == 'hardECH' or scenario == 'lightECH' or scenario == 'TLS1.2' or scenario == 'TLS1.3', 'Scenario is not correct!'
    EncryptedCH = True if scenario == 'hardECH' or scenario == 'lightECH'  else False
    EncryptedExtensions = True if scenario == 'TLS1.3' or scenario == 'hardECH' or scenario == 'lightECH' else False
    ThirdPacket = True if scenario == 'TLS1.2' else False
    change_len = True if scenario == 'hardECH' else False

    print('Initial Dataset Size: {}'.format(len(Dataset)))
    print('Only flows with the correct handshake remain')
    print('Payload conversion from string to uint8 array')
    if EncryptedCH:
        print('ClientHello payload without SNI and ALPN')
    if EncryptedExtensions:
        print('ServerHello payload without extensions marked as "enrypted" by IANA')
    if change_len:
        print('Length of ClientHello has changed')
    


    Data, labels = [], []
    
    Dataset.index = range(len(Dataset))
    for i in tqdm(Dataset.index):
         try:
            pkt1 = np.array(Dataset['PKT_1_payload'][i].split(','), dtype = np.uint8)
            pkt2 = np.array(Dataset['PKT_2_payload'][i].split(','), dtype = np.uint8)
            #Check that pkt1 == ClientHello and pkt2 == ServerHello
            if (pkt1[0] == 22 and pkt1[5] == 1) and (pkt2[0] == 22 and pkt2[5] == 2):
                if not EncryptedCH:
                    pkt1 = CH_payload_with_plastered_SNI(zero_random_utf(pkt1))
                pkt2 = zero_random_utf(pkt2)
                if EncryptedCH:
                    pkt1 = EnryptedCH_payload(pkt1, change_len = change_len, encrypted_extensions = ech_encrypted_extensions)
                if EncryptedExtensions:
                    pkt2 = SH_without_EE_payload(pkt2, EE_types)
                if ThirdPacket:
                    try: 
                        pkt3 = np.array(Dataset['PKT_3_payload'][i].split(','), dtype = np.uint8)
                    except:
                        pkt3 = np.array([], dtype = np.uint8)
                    Data.append([pkt1, pkt2, pkt3, Flow_Dataset[i], ipDst_Dataset[i]])
                    labels.append(Dataset[label_column][i])
                else:
                    if (get_msg_len(pkt1) == len(pkt1)) and (get_msg_len(pkt2) == len(pkt2)):
                        Data.append([pkt1, pkt2, Flow_Dataset[i], ipDst_Dataset[i]])
                        labels.append(Dataset[label_column][i])
         except:
             continue

    print('The resulting Dataset Size: {}'.format(len(Data)))
    return np.array(Data, dtype=object), np.array(labels, dtype=object)


def DataPayload_DataFlow_split(Data_mixed):
    DataPayload = pd.DataFrame(Data_mixed)[[0,1]].to_numpy()
    DataFlow_pr = pd.DataFrame(Data_mixed)[[2]].to_numpy()
    
    DataFlow = []
    for row in DataFlow_pr:
        DataFlow.append(row[0])
    
    DataFlow = np.array(DataFlow)
    
    return DataPayload, DataFlow


def DataPayload_DataFlow_DataIP_split(Data_mixed):
    DataPayload = pd.DataFrame(Data_mixed)[[0,1]].to_numpy()
    DataFlow_pr = pd.DataFrame(Data_mixed)[[2]].to_numpy()
    DataIP_pr = pd.DataFrame(Data_mixed)[[3]].to_numpy()

    DataFlow = []
    for row in DataFlow_pr:
        DataFlow.append(row[0])
    
    DataFlow = np.array(DataFlow)

    DataIP = []
    for row in DataIP_pr:
        DataIP.append(row[0])
    
    DataIP = np.array(DataIP)

    return DataPayload, DataFlow, DataIP

def DataPayload_DataFlow_DataBoF_DataIP_split(Data_mixed):
    DataPayload = pd.DataFrame(Data_mixed)[[0,1]].to_numpy()
    DataFlow_pr = pd.DataFrame(Data_mixed)[[2]].to_numpy()
    DataBoF_pr = pd.DataFrame(Data_mixed)[[3]].to_numpy()
    DataIP_pr = pd.DataFrame(Data_mixed)[[4]].to_numpy()

    DataFlow = []
    for row in DataFlow_pr:
        DataFlow.append(row[0])
    
    DataFlow = np.array(DataFlow)

    DataBoF = []
    for row in DataBoF_pr:
        DataBoF.append(row[0])
    
    DataBoF = np.array(DataBoF)

    DataIP = []
    for row in DataIP_pr:
        DataIP.append(row[0])
    
    DataIP = np.array(DataIP)

    return DataPayload, DataFlow, DataBoF, DataIP

def DataPayload_Flow_BoF_IP_Time_split(Data_mixed):
    # Convert mixed data to a DataFrame
    df = pd.DataFrame(Data_mixed)

    # Extract columns and convert to numpy arrays directly
    DataPayload = df.iloc[:, [0, 1]].to_numpy()
    DataFlow = df.iloc[:, 2].to_numpy()
    DataBoF = df.iloc[:, 3].to_numpy()
    DataIP = df.iloc[:, 4].to_numpy()
    DataTime = df.iloc[:, 5].to_numpy()

    DataFlow = np.array([arr.tolist() for arr in DataFlow])
    DataBoF = np.array([arr.tolist() for arr in DataBoF])
    DataIP = ['.'.join([str(i) for i in arr[np.argmax(arr > 0):]]) for arr in DataIP]
    DataTime = np.concatenate(DataTime)

    return DataPayload, DataFlow, DataBoF, DataIP, DataTime

def hybridC45_TLSfeatures(ClientHello_ServerHello: np.array) -> np.array:
    count=0
    payload = np.zeros(310, dtype = np.uint8) #306
    try: 
        ClientHello = ClientHello_ServerHello[0]
        
        CH_L_dict = {0: 123, 16: 125, 21: 127, 35: 129, 41: 131, 51: 133, 44: 135, 25:137}
        
        CH_D_dict = {3: {'position': 139, 'len':2},  15: {'position': 141, 'len':2}, 45: {'position': 143, 'len':2},
                    27: {'position': 145, 'len':4},  28: {'position': 149, 'len':4},  6: {'position': 153, 'len':4},
                    11: {'position': 157, 'len':4},  19: {'position': 161, 'len':4}, 20: {'position': 165, 'len':4},
                    58: {'position': 169, 'len':4},  43: {'position': 173, 'len':12}, 16: {'position':306, 'len':4},
                    10: {'position': 185, 'len':26}, 13: {'position': 211, 'len':26}}
        
        
        #RV, RL       #ML, MV       #SID len
        payload[0:4], payload[4:8], payload[8] = ClientHello[1:5], ClientHello[7:11], ClientHello[43]
        
        
        #Ciphers Len
        payload[9] = (ClientHello[44 + ClientHello[43]]*256 + ClientHello[45 + ClientHello[43]]) // 2
        
        L = 46 + ClientHello[43]
        r = payload[11]*2 if payload[11] <= 35 else 70
        #Cipher Suites
        payload[10: 10 + r] =  ClientHello[L:L + r]
        L += payload[9]*2 + ClientHello[L + payload[9]*2]+1
        #Extensions Len
        payload[80: 80+2] = ClientHello[L:L+2]
        L+=2

        type_position = 83 
        #extensions = ClientHello[L:]
        end_of_CH = payload[80]*256 + payload[81] + L
        
        while L < end_of_CH:
            ext_type = ClientHello[L]*256 + ClientHello[L+1]
            ext_len  = ClientHello[L+2]*256 + ClientHello[L+3]
            #ext_data = ClientHello[L+4:L+4+ext_len]
        
            payload[type_position: type_position + 2] = ClientHello[L:L+2]
            type_position+=2 if type_position < 123 else 0
            
            if ext_type in CH_L_dict:
                payload[CH_L_dict[ext_type]:CH_L_dict[ext_type] + 2] = ClientHello[L+2:L+4]
                L+= 4+ext_len
                continue
            elif ext_type in CH_D_dict:
                R = ext_len if ext_len <= CH_D_dict[ext_type]['len'] else CH_D_dict[ext_type]['len']
                payload[CH_D_dict[ext_type]['position']:CH_D_dict[ext_type]['position']+R] = ClientHello[L+4:L+4+R]
                L+= 4+ext_len
                continue
            else:
                L+= 4+ext_len

        #Ext_num
        payload[82] = (type_position-83)//2
            
        ServerHello = ClientHello_ServerHello[1]

        # if ServerHello[0]!=22:
        #     #TODO add packet number and QUIC version
        #     ServerHello = np.concatenate([np.array([2,0,0,0,0]), ServerHello])

        #Record version, Record Len, Handshake Type, Msg len, Msg Ver, SID len
        payload[237:241], payload[241:245], payload[245] = ServerHello[1:5], ServerHello[7:11], ServerHello[43]
        #cipher
        payload[246:248] = ServerHello[44 + ServerHello[43]:46 + ServerHello[43]]
        #ext_len
        payload[248:250] = ServerHello[47 + ServerHello[43]:49 + ServerHello[43]]

        '''
        SH_L_dict = {18: 290, 35: 292, 41: 294, 51: 296}

        SH_D_dict = {43: {'position': 298 , 'len':2},
                    51: {'position': 300, 'len':2},
                    11: {'position': 302, 'len':4}}
        '''
        SH_L_dict = {41: 294, 51: 296}

        SH_D_dict = {43: {'position': 299 , 'len':2},
                    51: {'position': 301, 'len':2}}
        
        
        L = 49 + ServerHello[43]
        end_of_SH = payload[248]*256 + payload[249] + L
        
        type_position = 250 

        while L < end_of_SH:
            ext_type = ServerHello[L]*256 + ServerHello[L+1]
            ext_len  = ServerHello[L+2]*256 + ServerHello[L+3]
            #ext_data = ServerHello[L+4:L+4+ext_len]
        
            payload[type_position: type_position + 2] = ServerHello[L:L+2]
            type_position+=2 if type_position < 290 else 0
            
            if ext_type in SH_L_dict:
                payload[SH_L_dict[ext_type]:SH_L_dict[ext_type] + 2] = ServerHello[L+2:L+4]
                L+= 4+ext_len
                continue
            elif ext_type in SH_D_dict:
                R = ext_len if ext_len <= SH_D_dict[ext_type]['len'] else SH_D_dict[ext_type]['len']
                payload[SH_D_dict[ext_type]['position']:SH_D_dict[ext_type]['position']+R] = ServerHello[L+4:L+4+R]
                L+= 4+ext_len
                continue
            else:
                L+= 4+ext_len

        #Ext_num
        payload[290] = (type_position-251)//2

    except: 
        count+=1
        
#     CH_SID_len = payload[8]
#     CH_Ciphers_len = payload[9]
#     CH_EXT_num = payload[82]
#     SH_SID_len = payload[245]
#     SH_EXT_num = payload[290]
#     Selected_SH_Cipher = payload[246:248]

    return np.array([payload[8], payload[9], payload[82], payload[245],
                                payload[246], payload[247], payload[290]])


def SH_selected_CipherSuite(ClientHello_ServerHello: np.array):
    payload = CH_and_SH_recomp(ClientHello_ServerHello)
    return tuple([hex(payload[246])[2:], hex(payload[247])[2:]])


def X_with_rand(Data: np.array, randmax = 256):
    Temp = pd.DataFrame(Data)
    Temp[Temp.shape[1]] = np.random.randint(randmax, size=len(Temp))
    return np.array(Temp)

def better_than_random_bytes(Data: np.array, y_samples, n_estimators = 150, random_state=27, max_features = 0.1,
                                      max_samples = None, max_depth = None):
    
    RF_temp = RandomForestClassifier(n_jobs=-5, n_estimators = n_estimators, random_state=random_state,
                                     max_features = max_features, max_samples = max_samples,
                                  max_depth = max_depth)
    
    RF_temp.fit(X_with_rand(Data),y_samples)
    Random_importance = RF_temp.feature_importances_[-1]
    better_random = []
    for i, item in enumerate(RF_temp.feature_importances_):
        if item > Random_importance:
            better_random.append(i)
            
    return better_random

def top_N(Data: np.array, y_samples, N = 250, n_estimators = 150, random_state=27, max_features = 0.1,
                                      max_samples = None, max_depth = None):
    
    RF_temp = RandomForestClassifier(n_jobs=-1, n_estimators = n_estimators, random_state=random_state,
                                     max_features = max_features, max_samples = max_samples,
                                  max_depth = max_depth)
    
    RF_temp.fit(Data,y_samples)
    top_features = []
    FI = sorted(list(RF_temp.feature_importances_), reverse=True)[N]
    for i, item in enumerate(RF_temp.feature_importances_):
        if item > FI:
            top_features.append(i)
            
    return top_features

def Data_with_selected_features(Data, selected_bytes):     
    return np.array(pd.DataFrame(Data)[selected_bytes])  

def RF_feature_selection_better_than_random(X_train, X_test, y_train, randmax = 256,
                                            n_estimators = 100, random_state = 1, max_features = 0.1, max_depth = None):
    
    better_than_rand_features = better_than_random_bytes(X_with_rand(X_train), y_train, n_estimators = n_estimators, random_state=random_state,
                                     max_features = max_features,  max_depth = max_depth)

    return Data_with_selected_features(X_train, better_than_rand_features), Data_with_selected_features(X_test, better_than_rand_features)


def TSPS_features_from_Table(TSPS_Table: pd.DataFrame)->dict:
    
        uplink_sizes = TSPS_Table.loc[TSPS_Table.isUplink == 1]['Size']
        downlink_sizes = TSPS_Table.loc[TSPS_Table.isUplink == 0]['Size']

        Pakets_unique = 6
        up_6_first = np.concatenate([uplink_sizes, np.zeros(Pakets_unique - len(uplink_sizes), dtype = np.uint8)]) if Pakets_unique - len(uplink_sizes) > 0 else uplink_sizes[:Pakets_unique]
        dw_6_first = np.concatenate([downlink_sizes, np.zeros(Pakets_unique - len(downlink_sizes), dtype = np.uint8)]) if Pakets_unique - len(downlink_sizes) > 0 else downlink_sizes[:Pakets_unique]

        uplink_times = np.array(TSPS_Table.loc[TSPS_Table.isUplink == 1]['Time'])[1:] - np.array(TSPS_Table.loc[TSPS_Table.isUplink == 1]['Time'])[:-1]
        downlink_times = np.array(TSPS_Table.loc[TSPS_Table.isUplink == 0]['Time'])[1:] - np.array(TSPS_Table.loc[TSPS_Table.isUplink == 0]['Time'])[:-1]

        uplink_times = uplink_times
        downlink_times = downlink_times

        if len(uplink_sizes) == 0:
            uplink_sizes = [0]
        if len(downlink_sizes) == 0:
            downlink_sizes = [0]

        if len(uplink_times) == 0:
            uplink_times = [0]
        if len(downlink_times) == 0:
            downlink_times = [0]
            
            
        main_features = {
                'UL_num': len(uplink_sizes), 'DL_num': len(downlink_sizes),
                'UL_PS_cumsum': sum(uplink_sizes), 'DL_PS_cumsum': sum(downlink_sizes),
                'UL_time': np.array(TSPS_Table.loc[TSPS_Table.isUplink == 1]['Time'])[-1] if len(np.array(TSPS_Table.loc[TSPS_Table.isUplink == 1]['Time'])) > 0 else 0,
                'DL_time': np.array(TSPS_Table.loc[TSPS_Table.isUplink == 0]['Time'])[-1] if len(np.array(TSPS_Table.loc[TSPS_Table.isUplink == 0]['Time'])) > 0 else 0,
                'UL_PS_max': max(uplink_sizes), 'UL_PS_min': min(uplink_sizes), 'UL_PS_ave': np.mean(uplink_sizes),                     'UL_PS_std': np.std(uplink_sizes), 
                'UL_PS_25th': np.percentile(uplink_sizes, 25), 'UL_PS_50th': np.percentile(uplink_sizes, 50),                           'UL_PS_75th': np.percentile(uplink_sizes, 75), 
                'DL_PS_max': max(downlink_sizes), 'DL_PS_min': min(downlink_sizes), 
                'DL_PS_ave': np.mean(downlink_sizes), 'DL_PS_std': np.std(downlink_sizes), 
                'DL_PS_25th': np.percentile(downlink_sizes, 25), 'DL_PS_50th': np.percentile(downlink_sizes, 50),                       'DL_PS_75th': np.percentile(downlink_sizes, 75),
                'UL_TI_max': max(uplink_times), 'UL_TI_min': min(uplink_times), 'UL_TI_ave': np.mean(uplink_times),                     'UL_TI_std': np.std(uplink_times), 
                'UL_TI_25th': np.percentile(uplink_times, 25), 'UL_TI_50th': np.percentile(uplink_times, 50),                           'UL_TI_75th': np.percentile(uplink_times, 75), 
                'DL_TI_max': max(downlink_times), 'DL_TI_min': min(downlink_times), 
                'DL_TI_ave': np.mean(downlink_times), 'DL_TI_std': np.std(downlink_times), 
                'DL_TI_25th': np.percentile(downlink_times, 25), 'DL_TI_50th': np.percentile(downlink_times, 50),                       'DL_TI_75th': np.percentile(downlink_times, 75)      
               }
            
        uplink_size_boxes = {'UL_PS_0-249': 0, 'UL_PS_250-499': 0, 
                             'UL_PS_500-749':0, 'UL_PS_750-999':0, 
                             'UL_PS_1000-1249':0, 'UL_PS_1250-MTU':0}
        
        downlink_size_boxes = {'DL_PS_0-249': 0, 'DL_PS_250-499': 0, 
                               'DL_PS_500-749':0, 'DL_PS_750-999':0, 
                               'DL_PS_1000-1249':0, 'DL_PS_1250-MTU':0}
        
        for ps in uplink_sizes:
            if ps < 250:
                uplink_size_boxes['UL_PS_0-249']+=1
            elif ps < 500:
                uplink_size_boxes['UL_PS_250-499']+=1
            elif ps < 750:
                uplink_size_boxes['UL_PS_500-749']+=1
            elif ps < 1000:
                uplink_size_boxes['UL_PS_750-999']+=1
            elif ps < 1250:
                uplink_size_boxes['UL_PS_1000-1249']+=1
            else:
                uplink_size_boxes['UL_PS_1250-MTU']+=1
                
        for ps in downlink_sizes:
            if ps < 250:
                downlink_size_boxes['DL_PS_0-249']+=1
            elif ps < 500:
                downlink_size_boxes['DL_PS_250-499']+=1
            elif ps < 750:
                downlink_size_boxes['DL_PS_500-749']+=1
            elif ps < 1000:
                downlink_size_boxes['DL_PS_750-999']+=1
            elif ps < 1250:
                downlink_size_boxes['DL_PS_1000-1249']+=1
            else:
                downlink_size_boxes['DL_PS_1250-MTU']+=1
                
        main_features.update(uplink_size_boxes)
        main_features.update(downlink_size_boxes)
        
        uplink_unique_sizes = {'UL_uPS_0': 0, 'UL_uPS_1': 0, 'UL_uPS_2': 0, 'UL_uPS_3': 0, 'UL_uPS_4': 0, 'UL_uPS_5':0}
        downlink_unique_sizes = {'DL_uPS_0': 0, 'DL_uPS_1': 0, 'DL_uPS_2': 0, 
                                 'DL_uPS_3': 0, 'DL_uPS_4': 0, 'DL_uPS_5':0}
        
        
        for i, value in enumerate(np.unique(uplink_sizes)[-6:]):
            uplink_unique_sizes['UL_uPS_'+str(i)] = value
        for i, value in enumerate(np.unique(downlink_sizes)[-6:]):
            downlink_unique_sizes['DL_uPS_'+str(i)] = value
          
        main_features.update(uplink_unique_sizes)
        main_features.update(downlink_unique_sizes)

        for i in range(Pakets_unique):
            main_features['UL_PS_{}'.format(i)] = up_6_first[i]
            main_features['DL_PS_{}'.format(i)] = dw_6_first[i]
            
        return  main_features

#this function TSPS_handshake_httpRec returns packets before sending application data

def TSPS_handshake_httpReq(Table_i):
    TSPS = pd.DataFrame( {'Time': [float(i) for i in Table_i['Time'].split(',')], 
                          'Size': [int(i) for i in Table_i['Size'].split(',')],
                          'isUplink': [int(i) for i in Table_i['isUplink'].split(',')]})
    
    TSPS = TSPS.loc[TSPS.Size > 0]
    mask = np.zeros(len(TSPS), dtype = bool)
    uplink = False
    uplink_cnt, downlink_cnt = 0,0
    for i, item in enumerate(TSPS['isUplink']):
        if item == 1 and uplink == False:
            uplink = True
            uplink_cnt+=1
        if item == 0 and uplink == True:
            downlink_cnt +=1
            uplink = False
        if downlink_cnt >=2:
            break
        else:
            mask[i] = True
    
    TSPS_handshake = TSPS.loc[mask]
    TSPS_handshake.index = range(len(TSPS_handshake))
    return TSPS_handshake

def flow_start_duration(Table_i, flow_start_function = TSPS_handshake_httpReq):
    if len(list(flow_start_function(Table_i)['Time'])) > 0:
        return list(flow_start_function(Table_i)['Time'])[-1]
    else:
        return 0

def TSPS_handshake(Table_i):
    TSPS = pd.DataFrame( {'Time': [float(i) for i in Table_i['Time'].split(',')], 
                          'Size': [int(i) for i in Table_i['Size'].split(',')],
                          'isUplink': [int(i) for i in Table_i['isUplink'].split(',')]})
    
    TSPS = TSPS.loc[TSPS.Size > 0]
    mask = np.zeros(len(TSPS), dtype = bool)
    uplink = False
    uplink_cnt, downlink_cnt = 0,0
    for i, item in enumerate(TSPS['isUplink']):
        if uplink_cnt >=2:
            break
        else:
            mask[i] = True
        if item == 1 and uplink == False:
            uplink = True
            uplink_cnt+=1
        if item == 0 and uplink == True:
            downlink_cnt +=1
            uplink = False
    
    TSPS_handshake = TSPS.loc[mask]
    TSPS_handshake.index = range(len(TSPS_handshake))
    return TSPS_handshake


def payload_data_TSPShandshake_exctraction(Dataset: pd.DataFrame, label_column: str, scenario = 'hardECH', handshake_ext = False,
                            ech_encrypted_extensions = set(list(range(41)) + [42] + list(range(44,51)) + list(range(52,61)) + [65281]), EE_types = set([0,1,10,14,15,16,19,20,28,32,39,42,55,56,57,58])):
    
    '''
    handshake_ext indicates limitation on data to use 
    False for only TLS handshake
    True for TLS handshake + client http req until first server app data
    '''

    assert scenario == 'hardECH' or scenario == 'lightECH' or scenario == 'TLS1.2' or scenario == 'TLS1.3', 'Scenario is not correct!'
    EncryptedCH = True if scenario == 'hardECH' or scenario == 'lightECH'  else False
    EncryptedExtensions = True if scenario == 'TLS1.3' or scenario == 'hardECH' or scenario == 'lightECH' else False
    change_len = True if scenario == 'hardECH' else False

    print('Initial Dataset Size: {}'.format(len(Dataset)))
    print('Only flows with the correct handshake remain')
    print('Payload conversion from string to uint8 array')
    if EncryptedCH:
        print('ClientHello payload without SNI and ALPN')
    if EncryptedExtensions:
        print('ServerHello payload without extensions marked as "enrypted" by IANA')
    if change_len:
        print('Length of ClientHello has changed')
    


    Data, labels = [], []
    
    Dataset.index = range(len(Dataset))
    for i in tqdm(Dataset.index):
         try:
            pkt1 = np.array(Dataset['PKT_1_payload'][i].split(','), dtype = np.uint8)
            pkt2 = np.array(Dataset['PKT_2_payload'][i].split(','), dtype = np.uint8)

            TSPS_hand = TSPS_handshake(Dataset.iloc[i]) if handshake_ext == False else TSPS_handshake_httpReq(Dataset.iloc[i])
            #Check that pkt1 == ClientHello and pkt2 == ServerHello
            if (pkt1[0] == 22 and pkt1[5] == 1) and (pkt2[0] == 22 and pkt2[5] == 2):
                if not EncryptedCH:
                    pkt1 = CH_payload_with_plastered_SNI(zero_random_utf(pkt1))
                pkt2 = zero_random_utf(pkt2)
                if EncryptedCH:
                    pkt1 = EnryptedCH_payload(pkt1, change_len = change_len, encrypted_extensions = ech_encrypted_extensions)
                if EncryptedExtensions:
                    pkt2 = SH_without_EE_payload(pkt2, EE_types)
                if (get_msg_len(pkt1) == len(pkt1)) and (get_msg_len(pkt2) == len(pkt2)):
                    TSPS_hand['Size'][0] = len(pkt1)
                    TSPS_hand['Size'][1] = len(pkt2)
                    TSPS_features = np.array(list(TSPS_features_from_Table(TSPS_hand).values()), dtype = float)
                    Data.append([pkt1, pkt2, TSPS_features])
                    labels.append(Dataset[label_column][i])
         except:
             continue

    print('The resulting Dataset Size: {}'.format(len(Data)))
    return np.array(Data, dtype=object), np.array(labels, dtype=object)


### Flow correlation code

def time_add(time: str, add_in_seconds:float):
    h, m, s = time.split('-')
    h = int(h)
    m = int(m)
    s = float(s)
    s+=add_in_seconds
    m+=s // 60
    s = s % 60
    h+=m // 60
    m = m % 60
    return '{}-{}-{}'.format(int(h),int(m),round(s,9))

def ms_between_times(time1:str, time2:str):
    h1, m1, s1 = time1.split('-')
    h2, m2, s2 = time2.split('-')
    return (int(h2)-int(h1))*60*60*1000 + (int(m2)-int(m1))*60*1000 + (float(s2)-float(s1))*1000 // 1

# def flow_start_duration(Table_i, flow_start_function = TSPS_handshake_httpReq):
#     return list(flow_start_function(Table_i)['Time'])[-1]

def is_time1_beq_time2(time1:str, time2:str):
    h1, m1, s1 = time1.split('-')
    h2, m2, s2 = time2.split('-')
    if int(h1) > int(h2):
        return True
    elif int(h1) < int(h2):
        return False
    elif int(h1) == int(h2):
        if int(m1) > int(m2):
            return True
        elif int(m1) < int(m2):
            return False
        elif int(m1) == int(m2):
            if float(s1) >= float(s2):
                return True
            elif float(s1) < float(s2):
                return False
            
def list_of_times_beq_time(list_of_times: list, time:str)->list:
    return [is_time1_beq_time2(time_i, time) for time_i in list_of_times]

def list_of_times_beq_time_vector(list_of_times: list, time:str)->list:
    return np.vectorize(is_time1_beq_time2)(list_of_times, time)

def bag_of_flows_legacy(df, i, prev_time_window = 5):
    mask = np.ones(len(df), dtype = bool)
    similar_origin = ['service', 'app', 'OS', 'data']
    for origin in similar_origin:
        mask = mask & (df[origin] == df.iloc[i][origin])
        
    df_i = df.loc[mask]
    new_mask = list_of_times_beq_time_vector(df_i['time_start'], time_add(df.iloc[i]['time_start'], -prev_time_window))
    new_mask = new_mask & np.invert(list_of_times_beq_time_vector(df_i['time_start'], time_add(df.iloc[i]['time_start'], flow_start_duration(df.iloc[i])/1000)))
    return df_i.loc[new_mask]  

def bag_of_flows(df, i, prev_time_window = 5):
    new_mask = (df['full_timestamp'] >= df.iloc[i]['full_timestamp'] - prev_time_window)
    new_mask = new_mask &  (df['full_timestamp'] <= df.iloc[i]['full_timestamp'] + flow_start_duration(df.iloc[i])/1000)
    return df.loc[new_mask]  


def bag_of_flows_TSPS(bag_of_flows: pd.DataFrame, target_flow_index: int, TLS_Interflow_features_flag = False)->pd.DataFrame:

    #in bag_of_flows pd.DataFrame indexes are kept as in original inital df with all flows
    #thus we could calculate 
    #BoF_end_time - absolute time when BoF caluclating should end - arrival time of last REQUEST before 1st APP data from server

    BoF_end_time = bag_of_flows.loc[target_flow_index]['full_timestamp']*1000 + flow_start_duration(bag_of_flows.loc[target_flow_index])

    TSPS = {'Time': np.array(bag_of_flows['full_timestamp'],dtype = float)*1000 - float(list(bag_of_flows['full_timestamp'])[0])*1000, 
            'UL_HndshSize':[], 
            'DL_HndshSize':[],
            'CH_CS_len': [],
            'SH_CS' :[]}
    
    bag_of_flows.index = range(len(bag_of_flows))
    for i in bag_of_flows.index:
        #relative_end_time is time when BoF ends in the system of current (i-th) flow
        #relative_end_time = BoF_end_time - flow_i_start_time 
        relative_end_time = BoF_end_time - bag_of_flows.iloc[i]['full_timestamp']
        TSPS_flow_handshReq = TSPS_handshake_httpReq(bag_of_flows.iloc[i])
        TSPS_flow_handshReq = TSPS_flow_handshReq.loc[TSPS_flow_handshReq['Time'] < relative_end_time]
        TSPS['UL_HndshSize'].append(sum(TSPS_flow_handshReq.loc[TSPS_flow_handshReq['isUplink'] == 1]['Size']))
        TSPS['DL_HndshSize'].append(sum(TSPS_flow_handshReq.loc[TSPS_flow_handshReq['isUplink'] == 0]['Size']))
        if TLS_Interflow_features_flag:
            TSPS['CH_CS_len'].append(bag_of_flows.iloc[i]['CH_SH_recomp'][9])
            TSPS['SH_CS'].append(bag_of_flows.iloc[i]['CH_SH_recomp'][246]*256 + bag_of_flows.iloc[i]['CH_SH_recomp'][247])
    
    if not TLS_Interflow_features_flag:
        del TSPS['CH_CS_len'], TSPS['SH_CS']

    TSPS = pd.DataFrame(TSPS)
    TSPS = TSPS.sort_values(by=['Time'])
    TSPS['Time'] = np.array(TSPS['Time'])+np.array(TSPS['Time'])[0]
    return TSPS

def ip_distinction(ipDst1: np.array, ipDst2: np.array) -> int:
    i = 0
    while ipDst1[i] == ipDst2[i]:
        i+=1
        if i == len(ipDst1):
            break 
    return len(ipDst1)-i


def ip_distinction_features(bag_of_flows_df: pd.DataFrame, ipDst_vector_i: list, bof_ip_distinction_max = 0)->dict:
    features = {}
    features['IP_dist_0'] = -1
    for i in range(1,17):
        features['IP_dist_' + str(i)] = 0
    
    assert 'ipDst_vector' in bag_of_flows_df.columns
    mask_bof_ip_distinction = []
    for IP in bag_of_flows_df['ipDst_vector']:
        ip_dist_j = ip_distinction(IP,ipDst_vector_i)
        features['IP_dist_' + str(ip_dist_j)]+=1
        mask_bof_ip_distinction.append(ip_dist_j <= bof_ip_distinction_max)
    return features, mask_bof_ip_distinction

def CH_SH_recomp_vector_distinction(CH_SH_recomp_1: np.array, CH_SH_recomp_2: np.array) -> int:
    return sum(np.array(CH_SH_recomp_1) != np.array(CH_SH_recomp_2))

def CH_SH_recomp_distinction_features(bag_of_flows_df: pd.DataFrame, CH_SH_recomp_i)->dict:
    features = {}
    features['CH_SH_dist_0'] = -1
    for i in range(1,26):
        features['CH_SH_dist_' + str(i)] = 0
    
    assert 'CH_SH_recomp' in bag_of_flows_df.columns
    
    for CH_SH_recomp_j in list(bag_of_flows_df['CH_SH_recomp']):
        dist = CH_SH_recomp_vector_distinction(CH_SH_recomp_j, CH_SH_recomp_i) // 2
        dist = 25 if dist > 25 else dist
        features['CH_SH_dist_' + str(dist)] +=1
    return features

    


def BoF_features(df: pd.DataFrame, i:int, prev_time_window: float = 5, 
        bof_ip_distinction_max = 0, TLS_Interflow_features_flag = False)->dict:

    try:

        ipDst_i = df.iloc[i]['ipDst']
        CH_SH_recomp_i = df.iloc[i]['CH_SH_recomp']
        BoF_df = bag_of_flows(df, i, prev_time_window = prev_time_window)
        assert 'ipDst_vector' in df.columns, 'create ipVector using ipVector function from ipDst column'
        ip_distinction_feat, mask_bof_ip_distinction = ip_distinction_features(BoF_df, df.iloc[i]['ipDst_vector'], bof_ip_distinction_max)
        CH_SH_recomp_distinction_feat = CH_SH_recomp_distinction_features(BoF_df, CH_SH_recomp_i)

        
        TSPS_BoF = bag_of_flows_TSPS(BoF_df.loc[mask_bof_ip_distinction], target_flow_index = i, TLS_Interflow_features_flag = TLS_Interflow_features_flag)

        Flows_unique = 10

        IAT = np.array(TSPS_BoF['Time'])[1:] - np.array(TSPS_BoF['Time'])[:-1]
        FS_UL = np.array(TSPS_BoF['UL_HndshSize'])
        FS_DL = np.array(TSPS_BoF['DL_HndshSize'])
        if TLS_Interflow_features_flag:
            CH_CS_Len_seq = np.array(TSPS_BoF['CH_CS_len'])
            SH_CS_seq = np.array(TSPS_BoF['SH_CS'])


        if len(IAT) == 0:
            IAT = [0]
        
        up_n_first = np.concatenate([FS_UL, np.zeros(Flows_unique - len(FS_UL), dtype = np.uint8)]) if Flows_unique - len(FS_UL) > 0 else FS_UL[:Flows_unique]
        dw_n_first = np.concatenate([FS_DL, np.zeros(Flows_unique - len(FS_DL), dtype = np.uint8)]) if Flows_unique - len(FS_DL) > 0 else FS_DL[:Flows_unique]

        if TLS_Interflow_features_flag:
            CH_CS_Len_seq_n_first = np.concatenate([CH_CS_Len_seq, np.zeros(Flows_unique - len(CH_CS_Len_seq), dtype = np.uint8)]) if Flows_unique - len(CH_CS_Len_seq) > 0 else CH_CS_Len_seq[:Flows_unique]    
            SH_CS_seq_n_first = np.concatenate([SH_CS_seq, np.zeros(Flows_unique - len(SH_CS_seq), dtype = np.uint8)]) if Flows_unique - len(SH_CS_seq) > 0 else SH_CS_seq[:Flows_unique]    
        

        main_features = {
                'Flows_num': len(TSPS_BoF),
                'UL_FS_cumsum': sum(FS_UL), 'DL_FS_cumsum': sum(FS_DL),
                'UL_PS_max': max(FS_UL), 'UL_PS_min': min(FS_UL), 'UL_PS_ave': np.mean(FS_UL),                     'UL_PS_std': np.std(FS_UL), 
                'UL_PS_25th': np.percentile(FS_UL, 25), 'UL_PS_50th': np.percentile(FS_UL, 50),                    'UL_PS_75th': np.percentile(FS_UL, 75), 
                'DL_PS_max': max(FS_DL), 'DL_PS_min': min(FS_DL), 
                'DL_PS_ave': np.mean(FS_DL), 'DL_PS_std': np.std(FS_DL), 
                'DL_PS_25th': np.percentile(FS_DL, 25), 'DL_PS_50th': np.percentile(FS_DL, 50),                       'DL_PS_75th': np.percentile(FS_DL, 75),
                'TI_max': max(IAT), 'TI_min': min(IAT), 'TI_ave': np.mean(IAT),                     'TI_std': np.std(IAT), 
                'TI_25th': np.percentile(IAT, 25), 'TI_50th': np.percentile(IAT, 50),                           'TI_75th': np.percentile(IAT, 75), 
                'TI_sum': sum(IAT)   
               }
        
        uplink_unique_sizes = {'UL_uPS_0': 0, 'UL_uPS_1': 0, 'UL_uPS_2': 0, 'UL_uPS_3': 0, 'UL_uPS_4': 0, 'UL_uPS_5':0}
        downlink_unique_sizes = {'DL_uPS_0': 0, 'DL_uPS_1': 0, 'DL_uPS_2': 0, 
                                 'DL_uPS_3': 0, 'DL_uPS_4': 0, 'DL_uPS_5':0}
        
        
        for i, value in enumerate(np.unique(FS_UL)[-6:]):
            uplink_unique_sizes['UL_uPS_'+str(i)] = value
        for i, value in enumerate(np.unique(FS_DL)[-6:]):
            downlink_unique_sizes['DL_uPS_'+str(i)] = value
          
        main_features.update(uplink_unique_sizes)
        main_features.update(downlink_unique_sizes)

        for i in range(Flows_unique):
            main_features['UL_PS_{}'.format(i)] = up_n_first[i]
            main_features['DL_PS_{}'.format(i)] = dw_n_first[i]
        
        main_features.update(ip_distinction_feat)
        main_features.update(CH_SH_recomp_distinction_feat)

        if TLS_Interflow_features_flag: 

            TLS_Interflow_features = {'CH_CS_len_cumsum': sum(CH_CS_Len_seq), 
                                    'CH_CS_len_max': max(FS_UL), 
                                    'CH_CS_len_min': min(CH_CS_Len_seq), 
                                    'CH_CS_len_ave': np.mean(CH_CS_Len_seq), 
                                    'CH_CS_len_std': np.std(CH_CS_Len_seq), 
                                    'CH_CS_len_25th': np.percentile(CH_CS_Len_seq, 25), 
                                    'CH_CS_len_50th': np.percentile(CH_CS_Len_seq, 50), 
                                    'CH_CS_len_75th': np.percentile(CH_CS_Len_seq, 75), 
                                    'SH_CS_cumsum': sum(SH_CS_seq),
                                    'SH_CS_max': max(SH_CS_seq), 
                                    'SH_CS_min': min(SH_CS_seq), 
                                    'SH_CS_ave': np.mean(SH_CS_seq), 
                                    'SH_CS_std': np.std(SH_CS_seq), 
                                    'SH_CS_25th': np.percentile(SH_CS_seq, 25),
                                    'SH_CS_50th': np.percentile(SH_CS_seq, 50),  
                                    'SH_CS_75th': np.percentile(SH_CS_seq, 75)}

            for i in range(Flows_unique):
                TLS_Interflow_features['CH_CS_len_{}'.format(i)] = CH_CS_Len_seq_n_first[i]
                TLS_Interflow_features['SH_CS_{}'.format(i)] = SH_CS_seq_n_first[i]
        
            main_features.update(TLS_Interflow_features)
            del TLS_Interflow_features, CH_CS_Len_seq, SH_CS_seq, CH_CS_Len_seq_n_first, SH_CS_seq_n_first
        
        del BoF_df, TSPS_BoF, IAT, FS_UL, FS_DL
        return  main_features
    except Exception as e:
        # Printing the exception
        print(f"An error occurred: {e}")
        return 0


def payload_data_TSPShandshake_BoF_exctraction(Dataset: pd.DataFrame, label_column: str, scenario = 'hardECH', handshake_ext = True,
                            ech_encrypted_extensions = set(list(range(41)) + [42] + list(range(44,51)) + list(range(52,61)) + [65281]), EE_types = set([0,1,10,14,15,16,19,20,28,32,39,42,55,56,57,58])):
    
    '''
    handshake_ext indicates limitation on data to use 
    False for only TLS handshake
    True for TLS handshake + client http req until first server app data
    '''


    assert scenario == 'hardECH' or scenario == 'lightECH' or scenario == 'TLS1.2' or scenario == 'TLS1.3', 'Scenario is not correct!'
    EncryptedCH = True if scenario == 'hardECH' or scenario == 'lightECH'  else False
    EncryptedExtensions = True if scenario == 'TLS1.3' or scenario == 'hardECH' or scenario == 'lightECH' else False
    change_len = True if scenario == 'hardECH' else False

    print('Initial Dataset Size: {}'.format(len(Dataset)))
    print('Only flows with the correct handshake remain')
    print('Payload conversion from string to uint8 array')
    if EncryptedCH:
        print('ClientHello payload without SNI and ALPN')
    if EncryptedExtensions:
        print('ServerHello payload without extensions marked as "enrypted" by IANA')
    if change_len:
        print('Length of ClientHello has changed')
    

    print('CH and SH recomposed payload vector creation')
    Dataset['CH_SH_recomp'] = CH_SH_recomp_vector_from_table(Dataset)
    Dataset['full_timestamp'] = Dataset.apply(get_full_timestamp_pd, axis=1)
     
    Data, labels = [], []
    
    Dataset.index = range(len(Dataset))
    for i in tqdm(Dataset.index):
        if Dataset.iloc[i]['protocol'] == 'TCP':
            try:
                pkt1 = np.array(Dataset['PKT_1_payload'][i].split(','), dtype = np.uint8)
                pkt2 = np.array(Dataset['PKT_2_payload'][i].split(','), dtype = np.uint8)

                TSPS_hand = TSPS_handshake(Dataset.iloc[i]) if handshake_ext == False else TSPS_handshake_httpReq(Dataset.iloc[i])
                #Check that pkt1 == ClientHello and pkt2 == ServerHello
                if (pkt1[0] == 22 and pkt1[5] == 1) and (pkt2[0] == 22 and pkt2[5] == 2):
                    if not EncryptedCH:
                        pkt1 = CH_payload_with_plastered_SNI(zero_random_utf(pkt1))
                    pkt2 = zero_random_utf(pkt2)
                    if EncryptedCH:
                        pkt1 = EnryptedCH_payload(pkt1, change_len = change_len, encrypted_extensions = ech_encrypted_extensions)
                    if EncryptedExtensions:
                        pkt2 = SH_without_EE_payload(pkt2, EE_types)
                    if (get_msg_len(pkt1) == len(pkt1)) and (get_msg_len(pkt2) == len(pkt2)):
                        TSPS_hand['Size'][0] = len(pkt1)
                        TSPS_hand['Size'][1] = len(pkt2)
                        TSPS_features = np.array(list(TSPS_features_from_Table(TSPS_hand).values()), dtype = float)
                        BoF_feat = np.array(list(BoF_features(Dataset,i).values()), dtype = float)
                        Data.append([pkt1, pkt2, TSPS_features, BoF_feat])
                        labels.append(Dataset[label_column][i])
            except:
                continue
        else:
            try:
                pkt1 = Dataset["PKT_1_payload"][i]
                pkt2 = Dataset["PKT_2_payload"][i]
                pkt1 = process_CH(pkt1, encrypted_extensions=ech_encrypted_extensions)
                pkt2 = process_SH(pkt2)
                TSPS_hand = TSPS_handshake(Dataset.iloc[i]) if handshake_ext == False else TSPS_handshake_httpReq(Dataset.iloc[i])
                if (get_msg_len_UDP(pkt1) == len(pkt1)) and (get_msg_len_UDP(pkt2) == len(pkt2)):
                    TSPS_hand['Size'][0] = len(pkt1)
                    TSPS_hand['Size'][1] = len(pkt2)
                    TSPS_features = np.array(list(TSPS_features_from_Table(TSPS_hand).values()), dtype = float)
                    BoF_feat = np.array(list(BoF_features(Dataset,i).values()), dtype = float)
                    Data.append([pkt1, pkt2, TSPS_features, BoF_feat])
                    labels.append(Dataset[label_column][i])
            except:
                try:
                    pkt1 = Dataset["PKT_1_payload"][i]
                    pkt2 = Dataset["PKT_3_payload"][i]
                    pkt1 = process_CH(pkt1, encrypted_extensions=ech_encrypted_extensions)
                    pkt2 = process_SH(pkt2)
                    TSPS_hand = TSPS_handshake(Dataset.iloc[i]) if handshake_ext == False else TSPS_handshake_httpReq(Dataset.iloc[i])
                    if (get_msg_len_UDP(pkt1) == len(pkt1)) and (get_msg_len_UDP(pkt2) == len(pkt2)):
                        TSPS_hand['Size'][0] = len(pkt1)
                        TSPS_hand['Size'][1] = len(pkt2)
                        TSPS_features = np.array(list(TSPS_features_from_Table(TSPS_hand).values()), dtype = float)
                        BoF_feat = np.array(list(BoF_features(Dataset,i).values()), dtype = float)
                        Data.append([pkt1, pkt2, TSPS_features, BoF_feat])
                        labels.append(Dataset[label_column][i])  
                except:
                    continue 
                continue

    print('The resulting Dataset Size: {}'.format(len(Data)))
    return np.array(Data, dtype=object), np.array(labels, dtype=object)



def get_full_timestamp(date_str, time_str):

    microseconds_str = time_str[:-3]  # This removes the last three digits (nanoseconds to microseconds)
    datetime_str = f"{date_str} {microseconds_str}"
    datetime_obj = datetime.strptime(datetime_str, "%d-%m-%Y %H-%M-%S.%f")
    nanoseconds = int(time_str[-3:])  # Get the last three digits as nanoseconds
    timestamp = datetime_obj.replace(tzinfo=timezone.utc).timestamp()
    additional_seconds = nanoseconds / 1e9
    full_timestamp = timestamp + additional_seconds

    #print("The full timestamp with nanoseconds is:", full_timestamp)
    return full_timestamp

def get_full_timestamp_pd(row, date_col = 'data' , time_col = 'time_start'):
    date_str = row[date_col]
    time_str = row[time_col]
    return get_full_timestamp(date_str, time_str)


def payload_data_TSPShandshake_BoF_IP_Timestamp_exctraction(Dataset: pd.DataFrame, 
                            label_column: str, 
                            scenario = 'hardECH', 
                            ip_dst_column = 'ipDst', 
                            prev_time_window: float = 5, 
                            bof_ip_distinction_max: int = 0,
                            handshake_ext = True,
                            TLS_Interflow_features_flag = False,
                            ech_encrypted_extensions = set(list(range(41)) + [42] + list(range(44,51)) + list(range(52,61)) + [65281]), 
                            EE_types = set([0,1,10,14,15,16,19,20,28,32,39,42,55,56,57,58])):
    
    '''
    handshake_ext indicates limitation on data to use 
    False for only TLS handshake
    True for TLS handshake + client http req until first server app data
    '''

    valid_scenarios = {'hardECH', 'lightECH', 'TLS1.2', 'TLS1.3'}
    assert scenario in valid_scenarios, 'Scenario is not correct!'
    EncryptedCH = scenario in {'hardECH', 'lightECH'}
    EncryptedExtensions = scenario in {'TLS1.3', 'hardECH', 'lightECH'}
    change_len = (scenario == 'hardECH')

    print('Initial Dataset Size: {}'.format(len(Dataset)))
    print('Only flows with the correct handshake remain')
    print('Payload conversion from string to uint8 array')
    if EncryptedCH:
        print('ClientHello payload without SNI and ALPN')
    if EncryptedExtensions:
        print('ServerHello payload without extensions marked as "enrypted" by IANA')
    if change_len:
        print('Length of ClientHello has changed')
    

    print('CH and SH recomposed payload vector creation')
    Dataset['CH_SH_recomp'] = CH_SH_recomp_vector_from_table(Dataset)

    ipDst_Dataset = ipVector(Dataset[ip_dst_column])
    Dataset['ipDst_vector'] = ipDst_Dataset
    Dataset['full_timestamp'] = Dataset.apply(get_full_timestamp_pd, axis=1)
    Dataset = Dataset.sort_values(by=['full_timestamp'])

    Data, labels = [], []
    
    Dataset.index = range(len(Dataset))
    for i in tqdm(Dataset.index):
        ipDSt_feat = ipDst_Dataset[i]
        timestamp_i = [float(Dataset.iloc[i]['full_timestamp'])]
        if Dataset.iloc[i]['protocol'] == 'TCP':
            try:
                pkt1 = np.array(Dataset['PKT_1_payload'][i].split(','), dtype = np.uint8)
                pkt2 = np.array(Dataset['PKT_2_payload'][i].split(','), dtype = np.uint8)

                TSPS_hand = TSPS_handshake(Dataset.iloc[i]) if handshake_ext == False else TSPS_handshake_httpReq(Dataset.iloc[i])
                #Check that pkt1 == ClientHello and pkt2 == ServerHello
                if (pkt1[0] == 22 and pkt1[5] == 1) and (pkt2[0] == 22 and pkt2[5] == 2):
                    if not EncryptedCH:
                        pkt1 = CH_payload_with_plastered_SNI(zero_random_utf(pkt1))
                    pkt2 = zero_random_utf(pkt2)
                    if EncryptedCH:
                        pkt1 = EnryptedCH_payload(pkt1, change_len = change_len, encrypted_extensions = ech_encrypted_extensions)
                    if EncryptedExtensions:
                        pkt2 = SH_without_EE_payload(pkt2, EE_types)
                    if (get_msg_len(pkt1) == len(pkt1)) and (get_msg_len(pkt2) == len(pkt2)):
                        TSPS_hand['Size'][0] = len(pkt1)
                        TSPS_hand['Size'][1] = len(pkt2)
                        TSPS_features = np.array(list(TSPS_features_from_Table(TSPS_hand).values()), dtype = float)
                        BoF_feat = np.array(list(BoF_features(Dataset, i, 
                                            prev_time_window, bof_ip_distinction_max, TLS_Interflow_features_flag).values()), dtype = float)
                        Data.append([pkt1, pkt2, TSPS_features, BoF_feat, ipDSt_feat, timestamp_i])
                        labels.append(Dataset[label_column][i])
            except:
                continue
        else:
            try:
                pkt1 = Dataset["PKT_1_payload"][i]
                pkt2 = Dataset["PKT_2_payload"][i]
                pkt1 = process_CH(pkt1, encrypted_extensions=ech_encrypted_extensions)
                pkt2 = process_SH(pkt2)
                TSPS_hand = TSPS_handshake(Dataset.iloc[i]) if handshake_ext == False else TSPS_handshake_httpReq(Dataset.iloc[i])
                if (get_msg_len_UDP(pkt1) == len(pkt1)) and (get_msg_len_UDP(pkt2) == len(pkt2)):
                    TSPS_hand['Size'][0] = len(pkt1)
                    TSPS_hand['Size'][1] = len(pkt2)
                    TSPS_features = np.array(list(TSPS_features_from_Table(TSPS_hand).values()), dtype = float)
                    BoF_feat = np.array(list(BoF_features(Dataset,i, 
                                prev_time_window, bof_ip_distinction_max, TLS_Interflow_features_flag).values()), dtype = float)
                    Data.append([pkt1, pkt2, TSPS_features, BoF_feat, ipDSt_feat, timestamp_i])
                    labels.append(Dataset[label_column][i])
            except:
                try:
                    pkt1 = Dataset["PKT_1_payload"][i]
                    pkt2 = Dataset["PKT_3_payload"][i]
                    pkt1 = process_CH(pkt1, encrypted_extensions=ech_encrypted_extensions)
                    pkt2 = process_SH(pkt2)
                    TSPS_hand = TSPS_handshake(Dataset.iloc[i]) if handshake_ext == False else TSPS_handshake_httpReq(Dataset.iloc[i])
                    if (get_msg_len_UDP(pkt1) == len(pkt1)) and (get_msg_len_UDP(pkt2) == len(pkt2)):
                        TSPS_hand['Size'][0] = len(pkt1)
                        TSPS_hand['Size'][1] = len(pkt2)
                        TSPS_features = np.array(list(TSPS_features_from_Table(TSPS_hand).values()), dtype = float)
                        BoF_feat = np.array(list(BoF_features(Dataset,i, 
                                    prev_time_window, bof_ip_distinction_max, TLS_Interflow_features_flag).values()), dtype = float)
                        Data.append([pkt1, pkt2, TSPS_features, BoF_feat, ipDSt_feat, timestamp_i])
                        labels.append(Dataset[label_column][i])  
                except:
                    continue 
                continue

    print('The resulting Dataset Size: {}'.format(len(Data)))
    del Dataset
    return np.array(Data, dtype=object), np.array(labels, dtype=object)

def CH_SH_recomp_vector_from_table(Dataset: pd.DataFrame, scenario = 'hardECH',
                            ech_encrypted_extensions = set(list(range(41)) + [42] + list(range(44,51)) + list(range(52,61)) + [65281]), EE_types = set([0,1,10,14,15,16,19,20,28,32,39,42,55,56,57,58])):
    
    '''
    handshake_ext indicates limitation on data to use 
    False for only TLS handshake
    True for TLS handshake + client http req until first server app data
    '''

    assert scenario == 'hardECH' or scenario == 'lightECH' or scenario == 'TLS1.2' or scenario == 'TLS1.3', 'Scenario is not correct!'
    EncryptedCH = True if scenario == 'hardECH' or scenario == 'lightECH'  else False
    EncryptedExtensions = True if scenario == 'TLS1.3' or scenario == 'hardECH' or scenario == 'lightECH' else False
    change_len = True if scenario == 'hardECH' else False
    


    CH_SH_recomp_list = []
    succ = 0
    
    Dataset.index = range(len(Dataset))
    for i in tqdm(Dataset.index):
        OK = False
        if Dataset.iloc[i]['protocol'] == 'TCP':
            try:
                pkt1 = np.array(Dataset['PKT_1_payload'][i].split(','), dtype = np.uint8)
                pkt2 = np.array(Dataset['PKT_2_payload'][i].split(','), dtype = np.uint8)
                #Check that pkt1 == ClientHello and pkt2 == ServerHello
                if (pkt1[0] == 22 and pkt1[5] == 1) and (pkt2[0] == 22 and pkt2[5] == 2):
                    if not EncryptedCH:
                        pkt1 = CH_payload_with_plastered_SNI(zero_random_utf(pkt1))
                    pkt2 = zero_random_utf(pkt2)
                    if EncryptedCH:
                        pkt1 = EnryptedCH_payload(pkt1, change_len = change_len, encrypted_extensions = ech_encrypted_extensions)
                    if EncryptedExtensions:
                        pkt2 = SH_without_EE_payload(pkt2, EE_types)
                    if (get_msg_len(pkt1) == len(pkt1)) and (get_msg_len(pkt2) == len(pkt2)):
                        CH_SH_recomp_list.append(list(CH_and_SH_recomp([pkt1, pkt2])))
                        succ+=1
                        OK = True
            except:
                OK = False
        else:
            try:
                pkt1 = Dataset["PKT_1_payload"][i]
                pkt2 = Dataset["PKT_2_payload"][i]
                pkt1 = process_CH(pkt1, encrypted_extensions=ech_encrypted_extensions)
                pkt2 = process_SH(pkt2)
                if (get_msg_len_UDP(pkt1) == len(pkt1)) and (get_msg_len_UDP(pkt2) == len(pkt2)):
                    CH_SH_recomp_list.append(list(CH_and_SH_recomp([pkt1, pkt2])))
                    succ+=1
                    OK = True
            except:
                try:
                    pkt1 = Dataset["PKT_1_payload"][i]
                    pkt2 = Dataset["PKT_3_payload"][i]
                    pkt1 = process_CH(pkt1, encrypted_extensions=ech_encrypted_extensions)
                    pkt2 = process_SH(pkt2)
                    if (get_msg_len_UDP(pkt1) == len(pkt1)) and (get_msg_len_UDP(pkt2) == len(pkt2)):
                        CH_SH_recomp_list.append(list(CH_and_SH_recomp([pkt1, pkt2])))
                        succ+=1
                        OK = True
                except:
                    OK = False
        if not OK:
            CH_SH_recomp_list.append(np.zeros(310, dtype = np.uint8))
    print(f'Percent of succesfully recomposed CH-SH pairs {succ*100/len(Dataset)}%')
    return CH_SH_recomp_list


def sorted_counted_dict(array: np.array):
    Dict = {}
    for item in set(array):
        Dict[item] = list(array).count(item)
    Sort_dict = {k: v for k, v in sorted(Dict.items(), key=lambda x: x[1], reverse=True)}
    return Sort_dict


def TLS_distinction_metric(Data, labels, fingerprint_type = 'RB', CH = True, SH = True)->pd.DataFrame:

    assert fingerprint_type == 'RB' or fingerprint_type == 'JA3'

    if fingerprint_type == 'RB':
        fingerprint= RBRF_data_without_GREASE(Data, njobs=10) 
    
    if fingerprint_type == 'JA3':
        fingerprint = JA3_data(Data, njobs=10) 

    hash_rb = {}
    hash_num = {}
    counter = 0

    for i, value in enumerate(fingerprint):

        if fingerprint_type == 'RB':
            if CH and SH:
                hash_v = hash(tuple(value))
            elif CH and not SH:
                hash_v = hash(tuple(value[:241]))
            elif not CH and SH:
                hash_v = hash(tuple(value[241:]))
            else:
                return False

        if fingerprint_type == 'JA3':
            if CH and SH:
                hash_v = hash(tuple(value))
            elif CH and not SH:
                hash_v = hash(tuple(value[:112]))
            elif not CH and SH:
                hash_v = hash(tuple(value[112:]))
            else:
                return False
            


        #hash_v = hash(tuple(value[241:303]))#SH
        #hash_v = hash(tuple(value[0:241]))#CH
        
        if not hash_v in hash_num:
            hash_num[hash_v] = counter
            counter+=1
            hash_rb[hash_num[hash_v]] = {}
            for service in set(labels):#AudioList+VideoList+ShortVideoList+['LiveVideo-YouTube', 'LiveVideo-Facebook', 'Web']:
                hash_rb[hash_num[hash_v]][service] = 0
            
        hash_rb[hash_num[hash_v]][labels[i]]+=1

    Hash_data = pd.DataFrame(hash_rb)
    fingerprint_metric = Hash_data.copy()
    for column in Hash_data.columns:
        fingerprint_metric[column] = np.array(Hash_data[column])/sum(np.array(Hash_data[column]))

    fingerprint_metric_cp = fingerprint_metric.copy()
    Hash_data_cp = Hash_data.copy()
    fingerprint_metric_cp.index = range(len(fingerprint_metric_cp))
    Hash_data_cp.index = range(len(Hash_data_cp))
    Metric_Final = []
    for index in Hash_data_cp.index :
        Metric_Final.append(np.dot(np.array(Hash_data_cp.iloc[index]),np.array(fingerprint_metric_cp.iloc[index]))/np.sum(Hash_data_cp.iloc[index]))
    Hash_data['Metric_finger_unique'] = np.array(Metric_Final)*100

    return Hash_data

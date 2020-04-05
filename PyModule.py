import pandas as pd
import numpy as np
from dateutil.parser import parse

# dictionary mapping official municipality twitter handles to the municipality name
mun_dict = {
    '@CityofCTAlerts' : 'Cape Town',
    '@CityPowerJhb' : 'Johannesburg',
    '@eThekwiniM' : 'eThekwini' ,
    '@EMMInfo' : 'Ekurhuleni',
    '@centlecutility' : 'Mangaung',
    '@NMBmunicipality' : 'Nelson Mandela Bay',
    '@CityTshwane' : 'Tshwane'
}


# dictionary of english stopwords
stop_words_dict = {
    'stopwords':[
        'where', 'done', 'if', 'before', 'll', 'very', 'keep', 'something', 'nothing', 'thereupon', 
        'may', 'why', 'â€™s', 'therefore', 'you', 'with', 'towards', 'make', 'really', 'few', 'former', 
        'during', 'mine', 'do', 'would', 'of', 'off', 'six', 'yourself', 'becoming', 'through', 
        'seeming', 'hence', 'us', 'anywhere', 'regarding', 'whole', 'down', 'seem', 'whereas', 'to', 
        'their', 'various', 'thereafter', 'â€˜d', 'above', 'put', 'sometime', 'moreover', 'whoever', 'although', 
        'at', 'four', 'each', 'among', 'whatever', 'any', 'anyhow', 'herein', 'become', 'last', 'between', 'still', 
        'was', 'almost', 'twelve', 'used', 'who', 'go', 'not', 'enough', 'well', 'â€™ve', 'might', 'see', 'whose', 
        'everywhere', 'yourselves', 'across', 'myself', 'further', 'did', 'then', 'is', 'except', 'up', 'take', 
        'became', 'however', 'many', 'thence', 'onto', 'â€˜m', 'my', 'own', 'must', 'wherein', 'elsewhere', 'behind', 
        'becomes', 'alone', 'due', 'being', 'neither', 'a', 'over', 'beside', 'fifteen', 'meanwhile', 'upon', 'next', 
        'forty', 'what', 'less', 'and', 'please', 'toward', 'about', 'below', 'hereafter', 'whether', 'yet', 'nor', 
        'against', 'whereupon', 'top', 'first', 'three', 'show', 'per', 'five', 'two', 'ourselves', 'whenever', 
        'get', 'thereby', 'noone', 'had', 'now', 'everyone', 'everything', 'nowhere', 'ca', 'though', 'least', 
        'so', 'both', 'otherwise', 'whereby', 'unless', 'somewhere', 'give', 'formerly', 'â€™d', 'under', 
        'while', 'empty', 'doing', 'besides', 'thus', 'this', 'anyone', 'its', 'after', 'bottom', 'call', 
        'nâ€™t', 'name', 'even', 'eleven', 'by', 'from', 'when', 'or', 'anyway', 'how', 'the', 'all', 
        'much', 'another', 'since', 'hundred', 'serious', 'â€˜ve', 'ever', 'out', 'full', 'themselves', 
        'been', 'in', "'d", 'wherever', 'part', 'someone', 'therein', 'can', 'seemed', 'hereby', 'others', 
        "'s", "'re", 'most', 'one', "n't", 'into', 'some', 'will', 'these', 'twenty', 'here', 'as', 'nobody', 
        'also', 'along', 'than', 'anything', 'he', 'there', 'does', 'we', 'â€™ll', 'latterly', 'are', 'ten', 
        'hers', 'should', 'they', 'â€˜s', 'either', 'am', 'be', 'perhaps', 'â€™re', 'only', 'namely', 'sixty', 
        'made', "'m", 'always', 'those', 'have', 'again', 'her', 'once', 'ours', 'herself', 'else', 'has', 'nine', 
        'more', 'sometimes', 'your', 'yours', 'that', 'around', 'his', 'indeed', 'mostly', 'cannot', 'â€˜ll', 'too', 
        'seems', 'â€™m', 'himself', 'latter', 'whither', 'amount', 'other', 'nevertheless', 'whom', 'for', 'somehow', 
        'beforehand', 'just', 'an', 'beyond', 'amongst', 'none', "'ve", 'say', 'via', 'but', 'often', 're', 'our', 
        'because', 'rather', 'using', 'without', 'throughout', 'on', 'she', 'never', 'eight', 'no', 'hereupon', 
        'them', 'whereafter', 'quite', 'which', 'move', 'thru', 'until', 'afterwards', 'fifty', 'i', 'itself', 'nâ€˜t',
        'him', 'could', 'front', 'within', 'â€˜re', 'back', 'such', 'already', 'several', 'side', 'whence', 'me', 
        'same', 'were', 'it', 'every', 'third', 'together'
    ]
}

def dictionary_of_metrics(items):
    """produce statistics metrics from a list and return a dictionary of the metrics"""
    
    metrics_dict = {}
    _mean = np.mean(items)
    metrics_dict['mean'] = round(_mean,2)
    
    _median = np.median(items)
    metrics_dict['median'] = round(_median,2)
    
    _std = np.std(items,ddof=1)
    metrics_dict['std'] = round(_std,2)
    
    _variance = np.var(items,axis=None)
    metrics_dict['var'] = round(_variance)
    
    _min = np.min(items)
    metrics_dict['min'] = round(_min,2)

    _max= np.max(items)
    metrics_dict['max'] = round(_max,2)
    
    return metrics_dict



### START FUNCTION
def five_num_summary(items):
    """five_num_summary produces a five number summary from a list and return a dictionary as a dictionary"""
   
    five_num_dict = {}

    _max = np.max(items)
    five_num_dict['max'] = _max
    
    _median = np.median(items)
    five_num_dict['median'] = _median
    
    _min = np.min(items)
    five_num_dict['min'] = _min
   
    first_quartile = np.percentile(items,25)
    five_num_dict['q1'] = first_quartile
    
    third_quartile = np.percentile(items,75)
    five_num_dict['q3'] = third_quartile
      
    return five_num_dict

### END FUNCTION

### START FUNCTION
def date_parser(dates):
    """date_parser produces a date string from a datatime string and return list of dates e.g From 'yyyy-mm-dd hh:mm:ss' to        'yyyy-mm-dd'"""

    from dateutil.parser import parse

    list_of_dates = []
    for _date in dates:
        formatted_date = parse(_date)
        list_of_dates.append(str(formatted_date.date()))

    return list_of_dates

### END FUNCTION

### START FUNCTION
def extract_municipality_hashtags(df):
    """extract_municipality_hashtags create two columns('municipality' and hashtags) and extract municipality related hashtags,and populated the columns"""
	
    import re as reg
    hashtag_list = []
    df['municipality'] = np.nan
    df['hashtags'] = np.nan

    for values in  mun_dict.values():
        for hashtag in range(len(twitter_df['Tweets'])):
            if df['Tweets'].str.contains('#')[hashtag]:
                hashtag_results = twitter_df['Tweets'][hashtag]
                hashtag_list = reg.findall(r'#\w+', res) 
                df['hashtags'][hashtag]=hashtag_list
                if reg.match(values,twitter_df['Tweets'][hashtag]):
                    df['municipality'][hashtag] = values
             


    return df

### END FUNCTION

### START FUNCTION
def number_of_tweets_per_day(df):
    """number_of_tweets_per_day calculates the number of tweets that were posted per day"""

    df = pd.read_csv(twitter_url).copy()
    tweets_df = pd.DataFrame()
    dates = df['Date'].to_list()
    date_results = []
    for _date in dates:
        dt = parse(_date)
        date_res.append(str(dt.date()))

    tweets_df['Tweets'] = df.pivot_table(index=date_results, aggfunc='size')
  

    return df

### END FUNCTION

### START FUNCTION
def word_splitter(df):
    """word_splitter splits the sentences in a dataframe's column into a list of the separate words"""
	
    tweets_list =df['Tweets'].to_list()
    split_list = []
    for tweets in tweets_list:
        split_list.append(tweets.split())
    df['Split Tweets'] = split_list
    
    return df

### END FUNCTION

### START FUNCTION
def stop_words_remover(df):
    """ stop_words_remover removes english stop words from a tweet"""

    tweets_list =df['Tweets'].to_list()
    new_list = []
    res = []
    for tweet_word in tweets_list:
        split_word=tweet_word.split()
        for word in split_word: 
            for values in stop_words_dict.values():
                if word not in values:
                    new_list.append(word)
        res.append(new_list)
        new_list = []
     df['Without Stop Words'] = res
     return df

### END FUNCTION
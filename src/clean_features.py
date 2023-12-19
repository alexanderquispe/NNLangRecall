
def read_data( df, omit_lexemes = False  ):
    
    import numpy as np
    import pandas as pd

    min_hl = 15.0 / ( 24 * 60 )
    max_hl = 274. 
    
    df[ 'p' ]          = df[ 'p_recall' ].apply( lambda x: np.clip( float( x ),  0.0001, .9999 ) )
    df[ 't' ]          = df[ 'delta' ].apply( lambda x: float( x ) / ( 60 * 60 * 24 ) )
    df[ 'h' ]          = df.apply( lambda row: np.clip( -row[ 't' ] / np.log2( row [ 'p' ] ), min_hl, max_hl ), axis = 1 )
    df[ 'lang' ]       = df.apply( lambda row: f"{ row[ 'ui_language' ] } -> { row[ 'learning_language' ] }", axis = 1 )
    df[ 'lexeme' ]     = df.apply( lambda row: f"{ row[ 'learning_language' ] }:{ row[ 'lexeme_string' ] }", axis = 1 )
    df[ 'right' ]      = df[ 'history_correct' ].astype( int )
    df[ 'wrong' ]      = df[ 'history_seen' ].astype( int ) - df[ 'right' ]    
    df[ 'right_this' ] = df[ 'session_correct' ].astype( int )
    df[ 'wrong_this' ] = df[ 'session_seen' ].astype( int ) - df[ 'right_this' ]  
    df[ 'right' ]      = df[ 'right' ].apply( lambda x: np.sqrt( 1 + x ) )
    df[ 'wrong' ]      = df[ 'wrong' ].apply( lambda x: np.sqrt( 1 + x ) )
    df[ 'bias' ]       = 1
    
    # Seconds to date
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
    
    # Function to convert time to decimal hours
    def time_to_decimal(time):
        return time.hour + time.minute / 60 + time.second / 3600

    # Apply the function to the 'datetime' column
    df['decimal_hours'] = df['datetime'].apply(lambda x: time_to_decimal(x))
    
    # create the year var
    df['date_only'] = pd.to_datetime(df['datetime']).dt.date

    if not omit_lexemes:
    
        lexeme_dummies = pd.get_dummies( df[ 'lexeme' ], dtype = float )
        lexeme_columns = lexeme_dummies.columns.to_list()
        df = pd.concat( [ df, lexeme_dummies ], axis = 1 )
        
        feature_vars = [ 'right', 'wrong', 'bias' ] + lexeme_columns
                         
    else:
                         
        feature_vars = [ 'right', 'wrong', 'bias' ]

    def categorize_time_of_day(hour):
        if 5 <= hour < 12:
            return 'morning'
        elif 12 <= hour < 17:
            return 'afternoon'
        elif 17 <= hour < 21:
            return 'evening'
        else:
            return 'night'

    # Apply the function to your DataFrame
    df['time_of_day'] = df['decimal_hours'].apply(categorize_time_of_day)

    # Get dummy variables
    time_of_day_dummies = pd.get_dummies(df['time_of_day'], prefix='time')

    # Join the dummies back to the original DataFrame or use them separately as needed
    df = df.join(time_of_day_dummies)
    
#     # Keep users either in training or testing
#     grouped = df.groupby('user_id')
    
#     # Create a list to store user IDs for training and testing
#     train_users = []
#     test_users = []

#     for user_id in grouped.groups.keys():
#         if np.random.rand() < 0.8:  # 80% chance
#             train_users.append(user_id)
#         else:
#             test_users.append(user_id)

#     trainset = df[df['user_id'].isin(train_users)]
#     testset = df[df['user_id'].isin(test_users)]

#     splitpoint = int(0.9 * len(df))
#     trainset   = df[ : splitpoint ]
#     testset    = df[ splitpoint : ]

    return df, feature_vars
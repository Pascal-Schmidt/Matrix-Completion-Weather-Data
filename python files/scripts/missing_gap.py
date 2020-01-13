def gap_above_below(df):
    for i in range(0, len(df)):
        
        counter_below = 1
        while(pd.isnull(df.loc[int(df["row"][i]) - counter_below,
                              df["col"][i]])):
            
            counter_below = counter_below + 1
            if (counter_below > 20):
                break
                
    
        counter_above = 1
        while(pd.isnull(df.loc[int(df["row"][i]) + counter_above,
                              df["col"][i]])):
            
            counter_below = counter_above + 1
            if (counter_above > 20):
                break
                
        df.loc[i, "below"] = counter_below - 1
        df.loc[i, "above"] = counter_above + 1
        
    return(df)
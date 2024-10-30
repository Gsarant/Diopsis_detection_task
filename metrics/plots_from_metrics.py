import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
import os 

def create_table_image(df, output_filename, title=None):
    # Read the CSV file
    #df = pd.read_csv(csv_file)
    def format_number(x):
        if isinstance(x, (int, float, np.number)):
            return f"{x:.3f}"
        return x

    # Apply formatting to all cells
    df_formatted = df.applymap(format_number)

    # Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=(12, len(df)*0.5+1))  # Adjust size based on data
    
    # Remove axes
    ax.axis('off')
    
    # Create the table
    table = ax.table(cellText=df_formatted.values,
                     colLabels=df.columns,
                     cellLoc='center',
                     loc='center')
    
    # Adjust table style
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)  # Adjust table size
    
    for key, cell in table.get_celld().items():
        if key[1] == 0:  # First column
            cell.set_width(0.4)  # Adjust this value to change the width of the first column
        else:
            cell.set_width(0.1)  # Adjust this value for other columns

    # Color the header row
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#4472C4')
        else:
            cell.set_facecolor('#E6F1FF' if row % 2 == 0 else 'white')
    
    # Add title if provided
    if title:
        plt.title(title, fontsize=16, fontweight='bold', pad=20)
    
    # Adjust layout and save
    plt.tight_layout()
    
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.close()


def create_average_plots_from_csv_files(csv_files,method,output_folder):
    if len(csv_files)==0:
        return
    dfs=[]
    for file in csv_files:
        df_tmp = pd.read_csv(file)
        dfs.append(df_tmp)
    
    avg_df = pd.DataFrame()
    for df in dfs:
        df = df.drop(['Num_of_images','map','map_75','map_small','map_medium','map_large','mar_1','mar_10','mar_100','mar_small','mar_medium','mar_large','map_per_class','map_per_class','mar_100_per_class','classes'], axis=1)    
        df = df.dropna(axis=1, how='all')
        mean_values = df.select_dtypes(include=[np.number]).mean()
        mean_row = pd.Series({
            'folder_name': f"Average ({'_'.join(df['folder_name'].iloc[0].split('_')[:-2])})",
            #'Num_of_images': df['Num_of_images'].sum(),
            **mean_values
        })
        mean_row=mean_row.to_frame(name='Average').transpose()
        
        avg_df = pd.concat([avg_df, mean_row], ignore_index=True)


    
    avg_df=avg_df.round(3)
    print(avg_df)
    plot_save_filename=f"{os.path.join(output_folder,f'method_{method}_average')}.png"
    create_table_image(avg_df, plot_save_filename, f'Average Metrics ({method})')
        

if __name__ == "__main__":
    output_folder = 'plots'
    folders=[
        '../kfolds/original_diopsis_data_set_train_val_yolo',
        '../kfolds/original_diopsis_data_set_train_val_synthetic_yolo',
        #,'../folds/original_diopsis_data_set_train_val_synthetic_coco'
        '../kfolds/original_diopsis_data_set_train_val_synthetic_cropted_yolo',
    ]
    all_csv_files=[]
    for folder in folders:
        csv_files = glob.glob(os.path.join(folder,'*.csv'))
        for csv_file in csv_files:
            df = pd.read_csv(csv_file)
            df = df.drop(['Num_of_images','map','map_75','map_small','map_medium','map_large','mar_1','mar_10','mar_100','mar_small','mar_medium','mar_large','map_per_class','map_per_class','mar_100_per_class','classes'], axis=1)
            df=df.round(3)
            plot_save_filename=f"{os.path.join(output_folder,f'{os.path.basename(folder)}_{os.path.basename(csv_file)}')}.png"
            title=f"Average Metrics ({os.path.basename(folder)} Method {os.path.basename(csv_file).split('_')[1]})"
            create_table_image(df,plot_save_filename,title=title)
            all_csv_files.append({'method':os.path.basename(csv_file).split('_')[1],'path':csv_file})

    methods=['default','sahi','ensemble']
    for method in methods:
        create_average_plots_from_csv_files([x['path'] for x in all_csv_files if x['method']==method],method,output_folder)
    
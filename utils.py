import osmnx as ox
import networkx as nx
import numpy as np


def create_time_shift_features(data, shift, label='Occupancy'):
    shifted_data = data[['ParkAddress', 'DateHour', label]]

    timedelta = data.iloc[1]['DateHour'] - data.iloc[0]['DateHour']

    shifted_values = shifted_data.groupby('ParkAddress')[label].shift(
        shift) * ((shifted_data['DateHour'] - (timedelta * shift)).isin(shifted_data['DateHour'].tolist())).replace(False, np.nan).astype(np.float32).values

    return shifted_values



def create_spatial_features(data, W, df_parks_info, neighbours):
    parks = list(df_parks_info['ParkAddress'])

    for i in range(1, neighbours + 1):
        data['AvailableStallsP' + str(i)] = np.nan


    for park in parks:
        neighbours_indexes = W[:, parks.index(park)].argsort()[:neighbours + 1][1:]
        i = 1

        for neighbour_index in neighbours_indexes:
            neighbour_park = parks[neighbour_index]

            n_stalls = df_parks_info[df_parks_info['ParkAddress'] == neighbour_park]['NumberOfStalls'].values[0]
            
            data.loc[data['ParkAddress'] == park, 'AvailableStallsP' + str(i)] = n_stalls - data[data['ParkAddress'] == neighbour_park]['OccupiedStalls'].values

            i += 1

    return data


def compute_parks_distance(coords, query, metric='route'):

        G = ox.graph_from_place(query, network_type='drive', simplify=True)

        distance_matrix = np.zeros(shape=(coords.shape[0], coords.shape[0]))

        for i in range (coords.shape[0]):
            for j in range (coords.shape[0]):    
                if i != j:
                    if metric == 'euclidean':
                        distance = np.linalg.norm(np.array(coords[i]) - np.array(coords[j]))
                        
                    if metric == 'route':
                        distance = nx.shortest_path_length(
                            G, 
                            ox.distance.nearest_nodes(G, coords[i][0], coords[i][1]), 
                            ox.distance.nearest_nodes(G, coords[j][0], coords[j][1]), 
                            weight='length'
                        )

                    distance_matrix[i, j] = distance

        return distance_matrix

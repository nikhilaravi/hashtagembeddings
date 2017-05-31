import nn as knn

find_k_nearest_words = knn.load_model()
top_k = 5

words_to_look_up = ['#love', '#food', '#party', '#datenight']
print ('finding nearest for ', words_to_look_up)
nearest_words_1 = find_k_nearest_words(words_to_look_up)
print('nearest words dict ', nearest_words_1)

words_to_look_up = ['#thoselegs', '#selfie', '#airport', '#flowers']
print ('finding nearest for ', words_to_look_up)
nearest_words_2 = find_k_nearest_words(words_to_look_up)
print('nearest words dict ', nearest_words_2)

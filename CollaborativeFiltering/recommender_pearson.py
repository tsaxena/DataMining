#dictionary of movie critics and their ratings of a small set of movies
# eucleadian distance 

from math import sqrt


critics ={ 
'Lisa Rose': {'Lady in the water' : 2.5, 
	      'Snakes on a Plane' : 3.5, 
              'Just my luck'      : 3.0, 
              'Superman Returns'  : 3.5, 
              'You me and Dupree' : 2.5, 
              'The Night Listener': 3.0
              },
'Gene Seymour': {'Lady in the water' : 3.0, 
                 'Snakes on a Plane' : 3.5, 
                 'Just my luck'      : 1.5, 
                 'Superman Returns'  : 5.0, 
                 'The Night Listener': 3.0,
                 'You me and Dupree' : 3.5
                },
'Michael Phillips': {'Lady in the water': 2.5, 
                     'Snakes on a Plane': 3.0, 
                     'Superman Returns' : 3.5, 
                     'The Night Listener': 4.0
                    },
'Claudia Puig': {'Snakes on a Plane'   : 3.5, 
                 'Just my luck'        : 3.0, 
                 'The Night Listener'  : 4.5,
                 'Superman Returns'    : 4.0, 
                 'You me and Dupree'   : 2.5
                },
'Mick LaSalle': {'Lady in the water' : 3.0, 
                 'Snakes on a Plane' : 4.0, 
                 'Just my luck'      : 2.0, 
                 'Superman Returns'  : 3.0, 
                 'The Night Listener': 3.0,
                 'You me and Dupree': 2.0
                },
'Jack Matthews': {'Lady in the water': 3.0, 
                  'Snakes on a Plane': 3.5, 
                  'Just my luck'     : 1.5, 
                  'Superman Returns' : 5.0, 
                  'The Night Listener': 3.0,
                  'You me and Dupree': 3.5},
'Toby': {         'Snakes on a Plane': 4.5, 
                  'Superman Returns' : 4.0, 
                  'You me and Dupree': 1.0},
}

def sim_distance(prefs, person1, person2):
   #get the list of shared movies that both rated.
   # create a dictionary consisting of common movies 
   # rated by both critics
   si = {}
   for item in prefs[person1]:
       if item in prefs[person2]:
          si[item] = 1
   #if they have no ratings in common, return 0
   if len(si)==0: return 0

   #Add up the squares of all the difference
   sum_of_squares = 0
   for item in si: 
     sum_of_squares += pow(prefs[person1][item] - prefs[person2][item], 2)
   
   print(len(si))
   return 1/(1+ sqrt(sum_of_squares)) 



def sim_pearson(prefs, person1, person2):
   #get the list of shared movies that both rated.
   # create a dictionary consisting of common movies 
   # rated by both critics
   si = {}
   for item in prefs[person1]:
       if item in prefs[person2]:
          si[item] = 1
   #if they have no ratings in common, return 0
   n = len(si)
   if n==0: return 0

   #Add up the squares of all the difference
   sum_xy = 0
   sum_x  = 0
   sum_y  = 0
   sum_x2 = 0
   sum_y2 = 0
   for item in si:
     x = prefs[person1][item]
     y = prefs[person2][item]
     sum_xy += x * y
     sum_x += x
     sum_y += y
     sum_x2 += pow(x, 2)
     sum_y2 += pow(y, 2)
   
   # numerator
   numerator = sum_xy - ((sum_x * sum_y)/n)
   # denominator 
   denominator = sqrt(sum_x2 - pow(sum_x, 2) / n) * sqrt(sum_y2 - pow(sum_y, 2) / n)
   
   if denominator==0: return 0

   return numerator/denominator

print(critics['Lisa Rose']['Lady in the water'])
print(sim_pearson(critics, 'Lisa Rose', 'Gene Seymour'))

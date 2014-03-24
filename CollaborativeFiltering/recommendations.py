#dictionary of movie critics and their ratings of a small set of movies
# eucleadian distance 

from math import sqrt

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

def topMatches(prefs, person, n = 5, similarity = sim_pearson):
    scores = [similarity(prefs, person, other) for other in prefs if other != person]
    scores.sort()
    scores.reverse()
    return scores[0:n]

def getRecommendations(prefs, person, similarity = sim_pearson):
    totals = {}
    simSums = {}
    for other in prefs: 
        if other == person: continue
        sim = similarity(prefs, person, other)

        #ignore scores of zero or lower
        if sim <=0: continue
    
        #ignore scores of zero or lower
        for item in prefs[other]:
            if item not in prefs[person] or prefs[person][item] == 0:
               totals.setdefault(item, 0)
               totals[item] += prefs[other][item]*sim
               #simSums 
               simSums.setdefault(item, 0)
               simSums[item] += sim

    #create the normalized list
    rankings = [(total/simSums[item], item) for item, total in totals.items()]
    
    #return the sorted list
    rankings.sort()
    rankings.reverse()
    return rankings

So this is what I told the Prof:
First stated the problem  that is the emerging entity problem that we are trying to solve
what the Autors of TempEL found was that 90% of the new entity found in TempEL where actually continual entity
so what TIGER did was for every mention they would enrich the data for that mention so now there is more information of that mention so it would help the new mention. It worked it and help reduce the error for new entity compared to TempEL.
There was a problem with it and that was as we enrich it the search would tend to move towards the data it was  enriched from and averaging the results so it worked till some extent
and RAED i havent read this paper  but i said her that this could solve the problem for new novel entity like COVID as it was unique.

I explained one more thing and that was the Prototye and the residual thing:
So for a new mention we will try to find the family in which this would belong to using the dot product or some similar stratigy so prototype is somewhat like a vector
after finding the family we could do something like we will use the residual and then we find the top similar entities  strip away the common parts and keep in mind the percentage of common details and if it is very very similar and even the different part seen to be like a addtion or edit to a entity then it is a continual entity and if the different parts are some waht talking about different thing and the common part is more that we just stipred of it then it is likely to be a sibling
i explained to something like this

and also i told her we could use all three models that is the above one, TIGER, RAED seeing the condition  lets say for covid we know its a novel entity so by the prototype we will get a family but that family would be loosely coupled and that how we could get that should we use RAED. For Example we got a new input as a FIFA 2026  by the prototype we will get it belongs to the Football family inside it belongs to FIFA family (we now the entity in these are highly coupled) and after that we use the residual logic and find the most similar entity in the family, so we would get something like FIFA 2022, FIFA 2018 , then we strip away the common part and keep a place holder like 80% of the information are similar and in the different part we see that what are the differences that is only the title name, teams, team memebers, winners etc so now we can tell that they are siblings and not  continual to FIFA 2022 or FIFA 2018 but as you see it was continual for FIFA. A simalar Example would be lets say i make a new football internation tornament then it would be in the football family and sibling to FIFA, UEFA etc 

Above are the things I understood
summary is we would use a graph based technique 
we would use multiple says to get the dataset like RAED TIGER and TempEL

Note: the above things migh be different from emerging-entity-linking-research-ideation.md that ok becaude It was a progress check, not a commitment to a design. any thing wrong in what i said the prof should not be considered further


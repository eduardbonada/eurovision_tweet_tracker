var eurotweetModule = angular.module('eurotweetModule', ["ngTable"]);

eurotweetModule.controller('eurotweetController', function($scope, $http){

    // when landing on the page, get the ranking
    $http.get('/api/ranking')
        .then(function(response) {
            
            var ranking = [];

            var pos = 1

            var num_tweets = 0

            Object.keys(response.data).forEach(function(country, index, array){
                ranking.push(
                    {
                        'position' : pos,
                        'name' : country, 
                        'positive' : response.data[country]['positive'],
                        'neutral' : response.data[country]['neutral'],
                        'negative' : response.data[country]['negative'],
                        'tweets' : response.data[country]['tweets'],
                        'positive_perc' : 100 * response.data[country]['positive_perc'],
                        'neutral_perc' : 100 * response.data[country]['neutral_perc'],
                        'negative_perc' : 100 * response.data[country]['negative_perc'],
                        'tweets_perc' : 100 * response.data[country]['tweets_perc'],
                        'score' : response.data[country]['predicted_score']
                    });
                
                num_tweets = num_tweets + response.data[country]['tweets'];
                
                pos = pos + 1;

            });

            $scope.ranking = ranking;
            $scope.num_countries = ranking.length;
            $scope.num_tweets = num_tweets;

            //console.log($scope.ranking);
        });

    // when landing on the page, get the ranking
    $http.get('/api/ranking_v2')
        .then(function(response) {
            
            var ranking = [];

            var pos = 1

            var num_tweets = 0

            Object.keys(response.data).forEach(function(country, index, array){
                ranking.push(
                    {
                        'position' : pos,
                        'name' : country, 
                        'positive' : response.data[country]['positive'],
                        'neutral' : response.data[country]['neutral'],
                        'negative' : response.data[country]['negative'],
                        'tweets' : response.data[country]['tweets'],
                        'positive_perc' : 100 * response.data[country]['positive_perc'],
                        'neutral_perc' : 100 * response.data[country]['neutral_perc'],
                        'negative_perc' : 100 * response.data[country]['negative_perc'],
                        'tweets_perc' : 100 * response.data[country]['tweets_perc'],
                        'score' : response.data[country]['predicted_score']
                    });
                
                num_tweets = num_tweets + response.data[country]['tweets'];
                
                pos = pos + 1;

            });

            $scope.ranking_v2 = ranking;
            $scope.num_countries_v2 = ranking.length;
            $scope.num_tweets_v2 = num_tweets;

            //console.log($scope.ranking);
        });
});
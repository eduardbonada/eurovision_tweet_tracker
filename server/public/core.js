var eurotweetModule = angular.module('eurotweetModule', []);

eurotweetModule.controller('eurotweetController', function($scope, $http){

    // when landing on the page, get the list of all hyped bands
    $http.get('/api/ranking')
        .then(function(response) {
            
            ranking = [];

            Object.keys(response.data).forEach(function(country, index, array){
                ranking.push(
                            {
                                'name' : country, 
                                'positive' : response.data[country]['positive'],
                                'neutral' : response.data[country]['neutral'],
                                'negative' : response.data[country]['negative'],
                                'tweets' : response.data[country]['tweets'],
                                'score' : Math.round(response.data[country]['predicted_score']*100)/100,
                            }
                );

            });

            $scope.ranking = ranking;

            //console.log($scope.ranking);
        });

});
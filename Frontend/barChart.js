var xValue = ['KNN', 'RF', 'GB', 'XGB'];

var yValue = [3234.809, 3520.05, 2266.071, 3478.123];
var yValue2 = [9678.056, 8997.641, 8672.925, 9360.442];

var testMAE = {
  x: xValue,
  y: yValue,
  type: 'bar',
  text: yValue.map(String),
  textposition: 'auto',
  hoverinfo: 'none',
  opacity: 0.5,
  name: 'testMAE',
  marker: {
    color: 'rgb(158,202,225)',
    line: {
      color: 'rgb(8,48,107)',
      width: 1.5
    }
  }
};

var testRMSE = {
  x: xValue,
  y: yValue2,
  type: 'bar',
  text: yValue2.map(String),
  textposition: 'auto',
  hoverinfo: 'none',
  name: 'testRMSE',
  marker: {
    color: 'rgba(58,200,225,.5)',
    line: {
      color: 'rgb(8,48,107)',
      width: 1.5
    }
  }
};

var data = [testMAE,testRMSE];

var layout = {
  title: 'Performance of different Machine Learning models'
};

Plotly.newPlot('myDiv', data, layout);

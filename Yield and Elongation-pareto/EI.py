import pandas as pd
from shapely.geometry import Polygon


pareto = [
(579, 27.8),
(579, 16.4),
(976, 16.4),
(976, 16.1),
(1028, 16.1),
(1028, 15.5),
(1111, 15.5),
(1111, 12.8),
(1192, 12.8),
(1192, 11.6534),
(1251.93, 11.6534)
]

fronto = pd.read_excel("/Users/wangping/Desktop/pythonProject/D-electron/Pareto/front.xlsx").iloc[:11, ]
front = fronto.iloc[:, 1:3]

area_list = []
for i in range(len(front)):
    point = (front["YS"][i], front["elongation"][i])


    pareto_polygon = [point]

    for i in range(len(pareto) - 1):
        x1, y1 = pareto[i]
        x2, y2 = pareto[i + 1]

        if y2 >= point[1] and y1 <= point[1] and x1<=point[0] and x2<=point[0]:
            pareto_polygon.append((x2, point[1]))

        if y2 >= point[1] and y1 >= point[1] and x1<=point[0] and x2<=point[0]:
            pareto_polygon.append(pareto[i])
        if x1 <= point[0] and x2 >= point[0]:
            pareto_polygon.append(pareto[i])
            pareto_polygon.append((point[0], y1))
            break

    #Create a polygon and calculate its area
    polygon = Polygon(pareto_polygon)
    #print(polygon)
    area = polygon.area
    area_list.append(area)


area_with_points = [(area_list[i], (front["YS"][i], front["elongation"][i])) for i in range(len(area_list))]
sorted_area_with_points = sorted(area_with_points, key=lambda x: x[0], reverse=True)
print(sorted_area_with_points)
sorted_points = [point for _, point in sorted_area_with_points]

df = pd.concat([fronto, pd.DataFrame(sorted_points, columns=["YS_order", "elongation_order"])], axis=1)

excel_filename = '/Users/wangping/Desktop/pythonProject/D-electron/Pareto/front_candidates.xlsx'
df.to_excel(excel_filename, index=False)

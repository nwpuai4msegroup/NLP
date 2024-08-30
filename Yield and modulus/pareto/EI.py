import pandas as pd
from shapely.geometry import Polygon


pareto = [
    (220, 20),
    (220, 45),
    (220, 47.8),
    (515, 47.8),
    (515, 53.2),
    (649, 53.2),
    (649, 66),
    (650, 66),
    (650, 69),
    (1027, 69),
    (1027, 71.1),
    (1062, 71.1),
    (1062, 74.9),
    (1152, 74.9),
    (1152, 110),
    (1250, 110),
    (1250, 115),
    (1539, 115),
    (1539, 124),
    (1785, 124),
    (1785, 130),
    (1847, 130)
]

fronto = pd.read_excel("/Users/wangping/Desktop/NLP/NLP-yield and modulus/pareto/front.xlsx")
front = fronto.iloc[:, 1:3]

area_list = []
for i in range(len(front)):
    point = (front["YS"][i], front["E"][i])


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


    polygon = Polygon(pareto_polygon)
    #print(polygon)
    area = polygon.area
    area_list.append(area)


area_with_points = [(area_list[i], (front["YS"][i], front["E"][i])) for i in range(len(area_list))]


sorted_area_with_points = sorted(area_with_points, key=lambda x: x[0], reverse=True)
print(sorted_area_with_points)
sorted_points = [point for _, point in sorted_area_with_points]

df = pd.concat([fronto, pd.DataFrame(sorted_points, columns=["YS_order", "E_order"])], axis=1)

excel_filename = '/Users/wangping/Desktop/NLP/NLP-yield and modulus/pareto/front.xlsx'
df.to_excel(excel_filename, index=False)

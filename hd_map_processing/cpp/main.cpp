#include "ticpp/ticpp.h"
#include "iostream"

int main () {
    try
    {
        ticpp::Document doc("/mnt/nextcloud/tum/Master/5. Semester/Guided Research/hd_map/2020_09_23_providentia_1_karte/2019-02-06_Providentia_A9.xodr" );
        doc.LoadFile();
        ticpp::Iterator< ticpp::Element > child( "road" );
        ticpp::Node* parent = doc.FirstChildElement();
        for ( child = child.begin( parent ); child != child.end(); child++ )
        {
            std::cout << child->Value() << std::endl;
        }
    }
    catch( ticpp::Exception& ex )
    {
        std::cout << ex.what();
    }

    return 0;
}
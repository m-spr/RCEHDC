-- MIT License

-- Copyright (c) 2024 m-spr

-- Permission is hereby granted, free of charge, to any person obtaining a copy
-- of this software and associated documentation files (the "Software"), to deal
-- in the Software without restriction, including without limitation the rights
-- to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
-- copies of the Software, and to permit persons to whom the Software is
-- furnished to do so, subject to the following conditions:

-- The above copyright notice and this permission notice shall be included in all
-- copies or substantial portions of the Software.

-- THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
-- IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
-- FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
-- AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
-- LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
-- OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
-- SOFTWARE.

LIBRARY IEEE;
USE IEEE.STD_LOGIC_1164.ALL;
USE IEEE.NUMERIC_STD.ALL;

ENTITY RSA IS
    GENERIC (
        inLen   : INTEGER := 8;  -- Bit width of popCounters
        d       : INTEGER := 8;  -- Number of popCounters
        z       : INTEGER := 0;  -- Zero padding to 2**
        logInNum: INTEGER := 3   -- MuxCell, ceilingLOG2(#popCounters)
    );   
    PORT (
        clk        : IN STD_LOGIC;
        rst        : IN STD_LOGIC;
        reg1Update : IN STD_LOGIC;
        reg1rst    : IN STD_LOGIC;
        reg2Update : IN STD_LOGIC;
        reg2rst    : IN STD_LOGIC;
        muxSel     : IN STD_LOGIC_VECTOR (logInNum DOWNTO 0);
        A          : IN STD_LOGIC_VECTOR (((d) * inLen) - 1 DOWNTO 0);   -- Cascade with enough 0 as input or inner signal!
        B          : OUT STD_LOGIC_VECTOR (inLen + logInNum - 1 DOWNTO 0)
    );
END ENTITY RSA;

ARCHITECTURE behavioral OF RSA IS

    COMPONENT recMux IS
        GENERIC (
            n      : INTEGER := 3;  -- MUX selection bit width (number of layers)
            lenPop : INTEGER := 8   -- Bit width of popCounters
        );
        PORT (
            sel  : IN STD_LOGIC_VECTOR (n DOWNTO 0);  -- Selection input
            din  : IN STD_LOGIC_VECTOR (((2**n) * inLen) - 1 DOWNTO 0);
            dout : OUT STD_LOGIC_VECTOR (inLen - 1 DOWNTO 0)
        ); 
    END COMPONENT;

    COMPONENT SeqAdder IS
        GENERIC (
            lenPop : INTEGER := 8  -- Bit width of popCounters
        );
        PORT (
            clk        : IN STD_LOGIC;
            rst        : IN STD_LOGIC;
            reg1Update : IN STD_LOGIC;
            reg1rst    : IN STD_LOGIC;
            reg2Update : IN STD_LOGIC;
            reg2rst    : IN STD_LOGIC;
            din        : IN STD_LOGIC_VECTOR (lenPop - 1 DOWNTO 0);
            dout       : OUT STD_LOGIC_VECTOR (lenPop - 1 DOWNTO 0)
        );
    END COMPONENT;

    CONSTANT zero_muxin : STD_LOGIC_VECTOR (((z + 1) * inLen) - 1 DOWNTO 0) := (OTHERS => '0');
    CONSTANT len        : INTEGER := inLen + logInNum;
    CONSTANT zero       : STD_LOGIC_VECTOR (logInNum - 1 DOWNTO 0) := (OTHERS => '0');

    SIGNAL muxOut : STD_LOGIC_VECTOR (inLen - 1 DOWNTO 0);
    SIGNAL muxIn  : STD_LOGIC_VECTOR (((2**logInNum) + 1) * inLen - 1 DOWNTO 0);
    SIGNAL add    : STD_LOGIC_VECTOR (len - 1 DOWNTO 0);

BEGIN
    
    -- Mux input preparation
    muxIn <= A & zero_muxin;

    -- Multiplexer instantiation with separate port mapping
    m : recMux 
        GENERIC MAP (
            n      => logInNum,    
            lenPop => inLen
        )
        PORT MAP (
            sel  => muxSel, 
            din  => muxIn((((2**logInNum) + 1) * inLen) - 1 DOWNTO inLen), 
            dout => muxOut
        );

    -- Addition preparation
    add <= zero & muxOut;

    -- Sequential Adder instantiation with separate port mapping
    addM : SeqAdder
        GENERIC MAP (
            lenPop => len
        )
        PORT MAP (
            clk        => clk, 
            rst        => rst, 
            reg1Update => reg1Update, 
            reg1rst    => reg1rst, 
            reg2Update => reg2Update, 
            reg2rst    => reg2rst, 
            din        => add, 
            dout       => B
        );

END ARCHITECTURE behavioral;
